"""ComposedModel: named encoders + heads + predictors behind one surface.

A model here is not one backbone; it is a composition declared in the
config:

    model:
      encoders:
        vision: {_target_: "encoder:resnext50"}       # modality: image
        text:   {_target_: "encoder:text_bow", ...}   # modality: text
      heads:
        proj128: {encoder: vision, _target_: "head:mlp", ...}
      predictors:                                     # optional (JEPA-style)
        pred: {inputs: [video], _target_: "predictor:mlp", ...}
      generative:                                     # optional
        vlm: {_target_: "generative:openai_chat", ...}  # text, image, video -> text

The vocabulary, one line each:
  encoder    one modality's tensor -> representation; declares modality and
             out_dim (see encoders/base.py)
  output     every encode() returns an EncoderOutput (encoders/base.py):
             pooled always, tokens/spatial/aux when the encoder offers
             them.  New fields are additive, so richer encoders never
             change a signature.
  head       small module on an encoder's pooled output (projector,
             classifier); names its encoder in the config
  predictor  representation(s) -> predicted representation; names the
             encoders it reads
  generative inputs of some modalities -> one output modality (a VLM
             answering in text); see generative/base.py.  A model may carry
             generative components and no encoders at all.
  capability what the composition can do, derived from its parts -- this is
             what evaluation compatibility is checked against, so the
             framework knows what a model can do rather than which class
             it is.

Losses, probes and the drivers reach the model ONLY through names:
encode(...) / apply_head(...) / predict(...) / generate(...).  Every checkpoint records the
name -> role mapping (export_contract), so a trained encoder can be reloaded
elsewhere by name without knowing how it was trained.

Note (from design review): slots -- the named groups of items inside one
batch -- are deliberately NEVER fused into a single encoder forward, because
that changes BatchNorm statistics and so changes model behavior.  What IS
shared is the per-slot, per-view encoder forward (`precompute_feats`): the
original trainers run the backbone once per crop and feed every head from
that result, so recomputing it per loss would double memory, compute and
BatchNorm updates.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from egolearn.core import keys as K
from egolearn.model.encoders.base import EncoderOutput

# fields an EncoderOutput can carry; heads declare which one they consume
OUTPUT_FIELDS = ("pooled", "tokens", "spatial")


class ComposedModel(nn.Module):
    def __init__(
        self,
        encoders: dict[str, nn.Module],
        heads: dict[str, dict] | None = None,
        predictors: dict[str, dict] | None = None,
        frozen: tuple[str, ...] | list[str] | set[str] = (),
        generative: dict[str, nn.Module] | None = None,
    ):
        """encoders: name -> encoder module (with modality/out_dim attrs)
        heads: name -> {"encoder": encoder_name, "field": output field
            (default "pooled"), "module": nn.Module}
        predictors: name -> {"inputs": [encoder_name, ...], "module": nn.Module}
        frozen: encoder names that are carried but never trained (a JEPA EMA
            target).  Frozen is a construction-time fact recorded here -- not
            re-derived from requires_grad later, so freezing the whole model
            for a linear probe cannot change how batches are routed.
        generative: name -> generative component (with input_modalities and
            output_modality attrs)
        """
        super().__init__()
        # Populated by build_model; manually assembled models need an explicit
        # construction config when loading a checkpoint for evaluation.
        self.construction_config: dict | None = None
        if not encoders and not generative:
            raise ValueError("ComposedModel needs at least one encoder or generative component")
        for name, enc in encoders.items():
            for attr in ("modality", "out_dim"):
                if not hasattr(enc, attr):
                    raise ValueError(
                        f"encoder {name!r} ({type(enc).__name__}) lacks the "
                        f"{attr!r} attribute -- see model/encoders/base.py "
                        "for the encoder contract"
                    )
        self.encoders = nn.ModuleDict(encoders)

        self._frozen = frozenset(frozen)
        unknown = self._frozen - set(self.encoders.keys())
        if unknown:
            raise ValueError(f"frozen names {sorted(unknown)} are not encoders")
        for name in self._frozen:
            self.encoders[name].requires_grad_(False)
            self.encoders[name].eval()

        # Routing table, fixed at construction: modality -> encoder name.
        # Frozen encoders exist to be addressed by name, so they only take
        # a modality when nothing trainable has it.
        self._route: dict[str, str] = {}
        by_modality: dict[str, list[str]] = {}
        for name, enc in self.encoders.items():
            by_modality.setdefault(str(enc.modality), []).append(name)
        for modality, names in by_modality.items():
            pool = [n for n in names if n not in self._frozen] or names
            if len(pool) > 1:
                raise ValueError(
                    f"several {modality!r} encoders would compete for modality "
                    f"routing ({pool}); keep one trainable encoder per modality "
                    "and mark extra copies (EMA targets) frozen: true"
                )
            self._route[modality] = pool[0]

        self._head_encoder: dict[str, str] = {}
        self._head_field: dict[str, str] = {}
        head_mods = {}
        for name, spec in (heads or {}).items():
            self._head_encoder[name] = spec["encoder"]
            self._head_field[name] = spec.get("field", "pooled")
            if self._head_field[name] not in OUTPUT_FIELDS:
                raise ValueError(
                    f"head {name!r}: field must be one of {OUTPUT_FIELDS}, "
                    f"got {self._head_field[name]!r}"
                )
            head_mods[name] = spec["module"]
        self.heads = nn.ModuleDict(head_mods)

        self._predictor_inputs: dict[str, list[str]] = {}
        pred_mods = {}
        for name, spec in (predictors or {}).items():
            self._predictor_inputs[name] = list(spec.get("inputs", []))
            pred_mods[name] = spec["module"]
        self.predictors = nn.ModuleDict(pred_mods)

        # Generative components route by output modality, like encoders by input.
        self._generative_route: dict[str, str] = {}
        self._generative_io: dict[str, dict] = {}
        for name, component in (generative or {}).items():
            for attr in ("input_modalities", "output_modality"):
                if not hasattr(component, attr):
                    raise ValueError(
                        f"generative component {name!r} ({type(component).__name__}) lacks "
                        f"the {attr!r} attribute -- see model/generative/base.py"
                    )
            output = str(component.output_modality)
            if output != "text":
                raise ValueError(
                    f"generative component {name!r} produces {output!r}; only text generation "
                    "is supported until a benchmark needs another output"
                )
            if output in self._generative_route:
                raise ValueError(
                    f"generative components {self._generative_route[output]!r} and {name!r} "
                    f"both produce {output!r}; keep one per output modality"
                )
            self._generative_route[output] = name
            declared: Any = component  # attributes checked above
            self._generative_io[name] = {
                "inputs": [str(m) for m in declared.input_modalities],
                "output": output,
            }
        self.generative = nn.ModuleDict(generative or {})

    # ---- what the composition offers ----------------------------------------
    def modalities(self) -> dict[str, str]:
        """encoder name -> modality."""
        return {name: str(enc.modality) for name, enc in self.encoders.items()}

    def frozen_names(self) -> frozenset[str]:
        """Encoders carried but never trained (`frozen: true`), e.g. a JEPA
        EMA target.  A construction-time fact -- see __init__."""
        return self._frozen

    def capabilities(self) -> frozenset[str]:
        """What this composition can do, derived from its parts.  Evaluation
        requirements are checked against this set at build time, so a probe
        that needs e.g. text representations fails to wire up against a
        vision-only model before any data is read."""
        modalities = set(self.modalities().values())
        caps = {f"{m}_representation" for m in modalities}
        if len(modalities) >= 2:
            caps.add(K.CAP_MULTIMODAL)
        if len(self.predictors) > 0:
            caps.add(K.CAP_PREDICTION)
        caps.update(f"{output}_generation" for output in self._generative_route)
        return frozenset(caps)

    def train(self, mode: bool = True) -> ComposedModel:
        """Standard nn.Module.train, except frozen encoders stay in eval
        mode: an EMA target with BatchNorm must not update its running
        statistics just because the rest of the model is training."""
        super().train(mode)
        for name in self._frozen:
            self.encoders[name].eval()
        return self

    # ---- encoders ------------------------------------------------------------
    def encoder_for(self, modality: str) -> nn.Module | None:
        """The encoder of a modality, or None.  The routing table is fixed
        at construction: frozen encoders (a JEPA EMA target) never take a
        modality that a trainable encoder has -- they exist to be addressed
        by name -- and ambiguous compositions are rejected in __init__, so
        this can never fail mid-run."""
        name = self._route.get(modality)
        return None if name is None else self.encoders[name]

    def encode(self, name: str, *inputs: torch.Tensor, **kw) -> EncoderOutput:
        """Run one encoder, addressed by its config name ("vision", "text")
        or by its modality ("image", "video").  Always returns an
        EncoderOutput; callers name the field they want (`.pooled`,
        `.tokens`, ...), so encoders can grow new fields without breaking
        anyone."""
        enc = self.encoders[name] if name in self.encoders else self.encoder_for(name)
        if enc is None:
            raise KeyError(f"model has no encoder named {name!r} and no {name!r} modality")
        return EncoderOutput.wrap(enc(*inputs, **kw))

    # ---- heads ---------------------------------------------------------------
    def head_names(self) -> list[str]:
        return sorted(self.heads.keys())

    def head_encoder(self, name: str) -> str:
        """The encoder whose output this head consumes."""
        return self._head_encoder[name]

    def head_field(self, name: str) -> str:
        """Which EncoderOutput field this head consumes (v1: pooled)."""
        return self._head_field[name]

    def apply_head(self, name: str, feats: torch.Tensor) -> torch.Tensor:
        """Heads eat tensors.  If you have an EncoderOutput, say which field
        you mean (normally `.pooled`) -- the choice should be visible at the
        call site, not made silently here."""
        if isinstance(feats, EncoderOutput):
            raise TypeError(
                f"apply_head({name!r}, ...) got an EncoderOutput; pick a field "
                "explicitly (e.g. pass out.pooled)"
            )
        return self.heads[name](feats)

    # ---- predictors ----------------------------------------------------------
    def predictor_names(self) -> list[str]:
        return sorted(self.predictors.keys())

    def predictor_inputs(self, name: str) -> list[str]:
        """The encoders whose representations this predictor reads."""
        return list(self._predictor_inputs[name])

    def predict(self, name: str, *reps: torch.Tensor, **kw) -> torch.Tensor:
        """Predictors eat tensors too; a JEPA-style predictor wants `.tokens`,
        an MLP wants `.pooled`, and that choice belongs to the caller."""
        for r in reps:
            if isinstance(r, EncoderOutput):
                raise TypeError(
                    f"predict({name!r}, ...) got an EncoderOutput; pick a field "
                    "explicitly (e.g. pass out.pooled or out.tokens)"
                )
        return self.predictors[name](*reps, **kw)

    # ---- generative components -------------------------------------------------
    def generate(self, name: str, **inputs):
        """Run one generative component, addressed by its config name ("vlm") or
        by its output modality ("text").  Inputs are batched lists (texts=,
        images=, videos=); see model/generative/base.py."""
        component: Any
        if name in self.generative:
            component = self.generative[name]
        elif name in self._generative_route:
            component = self.generative[self._generative_route[name]]
        else:
            raise KeyError(
                f"model has no generative component named {name!r} and none producing {name!r}"
            )
        return component.generate(**inputs)

    # ---- shared per-step features ---------------------------------------------
    def precompute_feats(self, batch: dict, device, skip: frozenset[str] = frozenset()) -> None:
        """Run the right encoder once per slot and view, storing the result in
        `slot["feats"]` as [B, V, D]; losses read these instead of re-running
        encoders.  Each slot declares what its views are (`slot["modality"]`,
        stamped by the composer from the slot spec) and is routed to the
        encoder of that modality.  Slots whose modality has no encoder are
        skipped (their loss, if any, computes what it needs itself), as are
        slots named in `skip` (the drivers pass the slots whose only declared
        consumers run their own forwards -- see losses/base.own_forward_slots)."""
        for name, slot in batch.items():
            if name in skip:
                continue
            views = slot.get(K.SLOT_VIEWS)
            if views is None:
                continue
            modality = slot.get(K.SLOT_MODALITY)
            if modality is None:
                raise KeyError(
                    f"slot {name!r} carries views but declares no modality; "
                    "whoever builds the batch must stamp slot['modality'] "
                    "(the composer does this from the slot spec)"
                )
            enc = self.encoder_for(modality)
            if enc is None:
                continue
            v = views.to(device, non_blocking=True)
            feats = [EncoderOutput.wrap(enc(v[:, i].contiguous())).pooled for i in range(v.size(1))]
            slot[K.SLOT_FEATS] = torch.stack(feats, dim=1)

    # ---- export contract: what a checkpoint records so parts reload by name ---
    def export_contract(self) -> dict:
        contract = {
            "encoders": self.modalities(),
            "frozen": sorted(self._frozen),
            "heads": {
                n: {"encoder": self._head_encoder[n], "field": self._head_field[n]}
                for n in self.heads
            },
            "predictors": dict(self._predictor_inputs),
            "capabilities": sorted(self.capabilities()),
            "preprocessing": "imagenet_norm",  # views normalize with ImageNet stats
        }
        if self.generative:
            # Only when present, so contracts of encoder-only checkpoints are unchanged.
            contract["generative"] = {
                name: {"inputs": list(io["inputs"]), "output": io["output"]}
                for name, io in self._generative_io.items()
            }
        return contract
