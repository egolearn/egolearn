"""Pipeline builder + build-time validation.

`build(cfg)` assembles the whole system from a plain config dict and then
runs `validate()`, which checks that every name the config references
actually exists (memories, tiers, views, heads, encoders, schemas, ...).
Combinations that fit wire up without extra code; combinations that don't
fail at build time with a clear error message, before any training happens.

The acceptance test for the whole framework (README): a method is fully
expressed by its config plus the method-novel components its project owns;
everything else comes from the core toolbox.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from egolearn.evaluation.probe import Probe
from egolearn.model.build import build_model

from .config.instantiate import instantiate
from .core import events as E
from .core import keys as K
from .core.context import Context
from .data.batching.composer import Composer
from .data.batching.recipe import Recipe
from .data.loading import Loader
from .data.sources.base import Stream
from .data.text import VocabTokenizer
from .data.views import ViewBank
from .memory.admission.base import EmbeddingProvider, EventBuilder
from .memory.labels import LabelStore
from .memory.table import TieredMemory
from .model.model import ComposedModel
from .training.driver import LockstepDriver
from .training.hooks import CheckpointHook, Hook, LogHook, ProbeHook, SnapshotHook, WandbHook
from .training.offline import OfflineDriver
from .training.solver import ProgressCosine, build_optimizer


class PipelineError(RuntimeError):
    pass


@dataclass
class Pipeline:
    ctx: Context
    stream: Stream
    views: ViewBank
    loader: Loader
    tokenizer: VocabTokenizer | None
    label_stores: dict[str, LabelStore]
    embedders: dict[str, EmbeddingProvider]
    admission: list[EventBuilder]
    memories: dict[str, TieredMemory]
    routes: dict[str, str]
    recipe: Recipe | None  # None in offline (IID) mode
    composer: Composer | None  # None in offline (IID) mode
    model: ComposedModel
    losses: list
    driver: LockstepDriver | OfflineDriver
    probes: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def mode(self) -> str:
        return "offline" if self.recipe is None else "streaming"

    def summary(self) -> str:
        lines = ["EgoLearn pipeline"]
        lines.append(f"  mode      : {self.mode}")
        lines.append(f"  stream    : {type(self.stream).__name__}")
        if isinstance(self.driver, OfflineDriver):
            d = self.driver
            lines.append(f"  batch     : {d.slot}(n={d.n}, views={d.views})")
        else:
            lines.append("  admission : " + ", ".join(type(b).__name__ for b in self.admission))
            for _name, mem in self.memories.items():
                lines.append(f"  memory    : {mem.stats()}")
            lines.append(f"  routes    : {self.routes}")
            assert self.recipe is not None
            lines.append(
                "  slots     : "
                + ", ".join(f"{s.name}(n={s.n},{s.unit})" for s in self.recipe.slots)
            )
        lines.append(
            "  losses    : " + ", ".join(f"{fn.name}(w={fn.weight:.2g})" for fn in self.losses)
        )
        lines.append(
            f"  model     : encoders {self.model.modalities()}  heads {self.model.head_names()}"
        )
        for w in self.warnings:
            lines.append(f"  WARNING   : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
def build(cfg: dict) -> Pipeline:
    # `_imports_`: modules whose import registers project-owned components
    # (admission components, losses, views, ...). This is how a config reaches
    # code that lives in projects/ -- the config only names a component, and
    # the registry finds it no matter which directory it lives in.
    import importlib

    for mod in cfg.get("_imports_", []):
        importlib.import_module(mod)

    ctx = Context(seed=cfg.get("seed", 1337), device=cfg.get("device", "cuda"))

    # training mode: streaming (memory + admission) or offline (a traditional
    # IID loop -- two separate loops, see training/offline.py)
    mode = (cfg.get("driver") or {}).get("mode", "streaming")
    if mode not in ("streaming", "offline"):
        raise PipelineError(f"driver.mode must be 'streaming' or 'offline', got {mode!r}")

    # data ------------------------------------------------------------------
    stream: Stream = instantiate(cfg["stream"])
    ctx.clock.total_observations = stream.total_observations()
    views = ViewBank(cfg.get("views", {}))
    workers = int((cfg.get("loader") or {}).get("workers", 0))
    if workers < 0:
        raise PipelineError(f"loader.workers must be >= 0, got {workers}")
    # workers re-run `_imports_` so project-registered views exist there too
    loader = Loader(stream, views, workers=workers, imports=list(cfg.get("_imports_", [])))

    tokenizer = None
    if cfg.get("tokenizer"):
        tk = dict(cfg["tokenizer"])
        tokenizer = VocabTokenizer.from_json(tk.pop("vocab_json"), **tk)

    # labels / embedders ------------------------------------------------------
    label_stores = {name: LabelStore(name) for name in cfg.get("label_stores", [])}
    embedders: dict[str, EmbeddingProvider] = {}
    for alias, spec in (cfg.get("embedders") or {}).items():
        emb = instantiate(spec)
        emb.bind(ctx, loader)  # model is attached below, once it exists
        embedders[alias] = emb

    # memory (built before admission: a builder may bind a memory) -----------
    if mode == "offline":
        for key in ("memories", "admission", "routes", "recipe"):
            if cfg.get(key):
                raise PipelineError(
                    "driver.mode=offline is the traditional IID loop -- no "
                    f"memory, no admission. Remove `{key}:` or set mode: streaming."
                )
    memories = {
        name: TieredMemory(name, **spec) for name, spec in (cfg.get("memories") or {}).items()
    }
    routes: dict[str, str] = dict(cfg.get("routes") or {})

    admission: list[EventBuilder] = []
    for spec in cfg.get("admission", []):
        spec = dict(spec)
        store_name = spec.pop("label_store", None)
        embedder_alias = spec.pop("embedder", None)
        memory_name = spec.pop("memory", None)
        builder: EventBuilder = instantiate(spec)
        store = label_stores.get(store_name) if store_name else None
        if store_name and store is None:
            raise PipelineError(
                f"admission {type(builder).__name__}: unknown label_store "
                f"{store_name!r} (declare it under `label_stores:`)"
            )
        embedder = embedders.get(embedder_alias) if embedder_alias else None
        if embedder_alias and embedder is None:
            raise PipelineError(
                f"admission {type(builder).__name__}: unknown embedder {embedder_alias!r}"
            )
        memory = memories.get(memory_name) if memory_name else None
        if memory_name and memory is None:
            raise PipelineError(
                f"admission {type(builder).__name__}: unknown memory {memory_name!r}"
            )
        builder.bind(ctx, loader=loader, embedder=embedder, label_store=store, memory=memory)
        admission.append(builder)

    # batching ----------------------------------------------------------------
    recipe = composer = None
    if mode == "streaming":
        recipe = Recipe.from_config(cfg["recipe"])
        composer = Composer(loader, tokenizer=tokenizer, label_stores=label_stores)

    # model ------------------------------------------------------------------
    build_warnings: list[str] = []
    try:
        model = build_model(
            cfg["model"],
            vocab_size=tokenizer.vocab_size if tokenizer is not None else None,
            imports=cfg.get("_imports_", []),
        )
    except (TypeError, ValueError) as e:
        raise PipelineError(str(e)) from e

    # give learner embedders access to the model
    for emb in embedders.values():
        emb.model = model

    # losses / solver ----------------------------------------------------------
    losses = [instantiate(spec) for spec in cfg.get("losses", [])]
    scfg = dict(cfg["solver"])
    schedule_cfg = scfg.pop("schedule", None)
    optimizer = build_optimizer(
        model,
        scfg["groups"],
        opt=scfg.get("opt", "adamw"),
        momentum=scfg.get("momentum", 0.9),
        warnings=build_warnings,
        betas=scfg.get("betas", (0.9, 0.999)),
    )
    schedule = ProgressCosine(**schedule_cfg) if schedule_cfg else None

    # hooks / probes -------------------------------------------------------------
    run_dir = cfg.get("run_dir", "runs/default")
    hcfg = cfg.get("hooks", {})
    hooks: list[Hook] = [
        LogHook(every=hcfg.get("log_every", 50), timing=hcfg.get("log_timing", True))
    ]
    if hcfg.get("checkpoint_every"):
        hooks.append(CheckpointHook(run_dir, every_steps=hcfg["checkpoint_every"]))
    probes = []
    for spec in cfg.get("probes") or []:
        if str(spec.get("_target_", "")).startswith("benchmark:"):
            raise PipelineError(
                "standalone benchmarks cannot be training probes; use egolearn.evaluate"
            )
        probe = instantiate(spec)
        if not isinstance(probe, Probe):
            raise PipelineError("training probes must implement Probe; use egolearn.evaluate")
        probes.append(probe)
    if probes:
        hooks.append(ProbeHook(probes, every_steps=hcfg.get("probe_every", 500)))
    if hcfg.get("snapshot_milestones") or hcfg.get("snapshot_every"):
        hooks.append(
            SnapshotHook(
                run_dir,
                milestones=hcfg.get("snapshot_milestones") or [],
                every_steps=hcfg.get("snapshot_every", 0),
            )
        )
    wcfg = cfg.get("wandb") or {}
    if wcfg.get("mode", "disabled") != "disabled":
        hooks.append(
            WandbHook(
                entity=wcfg.get("entity"),
                project=wcfg.get("project"),
                mode=wcfg["mode"],
                run_name=wcfg.get("run_name"),
                config=cfg,
                every=hcfg.get("log_every", 50),
            )
        )

    dcfg = cfg["driver"]
    prefetch = bool(dcfg.get("prefetch", False))
    if prefetch:
        if workers <= 0:
            raise PipelineError(
                "driver.prefetch composes the next batch while the current "
                "one trains, which needs loader.workers >= 1 (the in-process "
                "load path manipulates global RNG state and cannot run "
                "concurrently with training)"
            )
        mutators = [fn.name for fn in losses if fn.mutates_sampling_state]
        mutators += [type(h).__name__ for h in hooks if h.mutates_sampling_state]
        if mutators:
            raise PipelineError(
                f"driver.prefetch=true, but {mutators} declare "
                "mutates_sampling_state: they change what the composer would "
                "sample between steps, so batches cannot be composed ahead"
            )

    driver: LockstepDriver | OfflineDriver
    if mode == "offline":
        driver = OfflineDriver(
            stream=stream,
            loader=loader,
            model=model,
            losses=losses,
            optimizer=optimizer,
            schedule=schedule,
            ctx=ctx,
            hooks=hooks,
            epochs=dcfg.get("epochs"),
            max_steps=dcfg.get("max_steps"),
            subsample=dcfg.get("subsample", 1),
            grad_accumulation=dcfg.get("grad_accumulation", 1),
            prefetch=prefetch,
            **(dcfg.get("batch") or {}),
        )
    else:
        assert recipe is not None and composer is not None
        driver = LockstepDriver(
            stream=stream,
            admission=admission,
            memories=memories,
            routes=routes,
            recipe=recipe,
            composer=composer,
            model=model,
            losses=losses,
            optimizer=optimizer,
            schedule=schedule,
            ctx=ctx,
            advance=dcfg.get("advance"),
            rate=dcfg.get("rate"),
            warmup=dcfg.get("warmup"),
            hooks=hooks,
            max_steps=dcfg.get("max_steps"),
            grad_accumulation=dcfg.get("grad_accumulation", 1),
            prefetch=prefetch,
        )

    pipe = Pipeline(
        ctx=ctx,
        stream=stream,
        views=views,
        loader=loader,
        tokenizer=tokenizer,
        label_stores=label_stores,
        embedders=embedders,
        admission=admission,
        memories=memories,
        routes=routes,
        recipe=recipe,
        composer=composer,
        model=model,
        losses=losses,
        probes=probes,
        driver=driver,
    )
    pipe.warnings.extend(build_warnings)
    validate(pipe)
    return pipe


# ---------------------------------------------------------------------------
def validate(pipe: Pipeline) -> None:
    """Check that every name the config references actually exists, at every
    boundary between pipeline stages (routes, slots, views, heads, encoders, ...).
    """
    errs: list[str] = []

    if isinstance(pipe.driver, OfflineDriver):
        _validate_offline(pipe, pipe.driver, errs)
        _validate_losses(pipe, {pipe.driver.slot}, errs)
        _validate_model_consumers(pipe, errs)
        if errs:
            raise PipelineError("pipeline validation failed:\n  - " + "\n  - ".join(errs))
        return
    assert pipe.recipe is not None

    # admission emissions vs routes
    emitted: set[str] = set()
    for b in pipe.admission:
        emitted.update(b.emits)
    for schema_name, mem in pipe.routes.items():
        E.schema(schema_name)  # raises on unknown schema
        if mem not in pipe.memories:
            errs.append(f"route {schema_name} -> {mem!r}: unknown memory")
        if schema_name not in emitted:
            pipe.warnings.append(f"route for {schema_name!r} but no admission component emits it")
    for schema_name in emitted:
        if schema_name not in pipe.routes:
            pipe.warnings.append(
                f"admission emits {schema_name!r} but no route stores it (events discarded)"
            )

    # slots
    def check_recipe(recipe: Recipe, tag: str):
        for slot in recipe.slots:
            src = slot.source
            if src["kind"] in ("tier", "mix"):
                mem_name = src.get("memory")
                mem = pipe.memories.get(mem_name) if mem_name else None
                if mem is None:
                    errs.append(f"{tag} slot {slot.name!r}: unknown memory {src.get('memory')!r}")
                elif src["kind"] == "tier" and src.get("tier") not in mem.tiers:
                    errs.append(
                        f"{tag} slot {slot.name!r}: memory "
                        f"{src['memory']!r} has no tier {src.get('tier')!r}"
                    )
                elif src["kind"] == "mix":
                    for t in src.get("fractions", {}):
                        if t not in mem.tiers:
                            errs.append(f"{tag} slot {slot.name!r}: mix names unknown tier {t!r}")
            elif src["kind"] == "stream_head":
                E.schema(src["schema"])
            for v in slot.views:
                if v not in pipe.views.names():
                    errs.append(
                        f"{tag} slot {slot.name!r}: unknown view "
                        f"{v!r} (declared: {pipe.views.names()})"
                    )
            if slot.text and pipe.tokenizer is None:
                errs.append(f"{tag} slot {slot.name!r}: text=true but no tokenizer configured")
            if slot.modality not in pipe.model.modalities().values():
                pipe.warnings.append(
                    f"{tag} slot {slot.name!r} declares modality "
                    f"{slot.modality!r} but no encoder has it; the slot gets "
                    "no precomputed feats, so its loss must work from raw views"
                )
            if slot.labels_from and slot.labels_from not in pipe.label_stores:
                errs.append(f"{tag} slot {slot.name!r}: unknown label store {slot.labels_from!r}")
        if recipe.fallback:
            check_recipe(recipe.fallback, tag + ".fallback")

    check_recipe(pipe.recipe, "recipe")

    _validate_losses(pipe, set(pipe.recipe.slot_names()), errs)
    _validate_model_consumers(pipe, errs)

    # driver wiring
    assert isinstance(pipe.driver, LockstepDriver)
    warm = pipe.driver.warmup
    if "tier_full" in warm:
        mem_name, _, tier = warm["tier_full"].partition(".")
        if mem_name not in pipe.memories or tier not in pipe.memories[mem_name].tiers:
            errs.append(f"driver.warmup.tier_full={warm['tier_full']!r}: no such memory.tier")
    adv = pipe.driver.advance
    if adv["kind"] == "events":
        E.schema(adv["schema"])
        if adv["schema"] not in emitted:
            errs.append(
                "driver.advance waits for {!r} but no admission component "
                "emits it (the run would stall)".format(adv["schema"])
            )

    if errs:
        raise PipelineError("pipeline validation failed:\n  - " + "\n  - ".join(errs))


def _validate_model_consumers(pipe: Pipeline, errs: list[str]) -> None:
    """Model <-> evaluation compatibility, plus predictor wiring.

    A probe declares which capabilities the model must have (see
    evaluation/probe.py); the model derives its capabilities from its
    components.  Comparing the two here means an incompatible model x
    evaluation pair fails at build time with one message, instead of each
    pair needing an adapter."""
    caps = pipe.model.capabilities()
    for probe in pipe.probes:
        for c in getattr(probe, "requires_capabilities", ()):
            if c in caps:
                continue
            if c in K.KNOWN_CAPABILITIES or c.endswith("_representation"):
                errs.append(
                    f"probe {probe.name!r} needs capability {c!r}; model has {sorted(caps)}"
                )
            else:
                errs.append(
                    f"probe {probe.name!r} requires {c!r}, which is not a "
                    f"capability name (known: {sorted(K.KNOWN_CAPABILITIES)})"
                )
    encoder_names = set(pipe.model.encoders.keys())
    for hname in pipe.model.head_names():
        if pipe.model.head_encoder(hname) not in encoder_names:
            errs.append(
                f"head {hname!r} reads encoder {pipe.model.head_encoder(hname)!r}; "
                f"model has {sorted(encoder_names)}"
            )
    for pname in pipe.model.predictor_names():
        for ename in pipe.model.predictor_inputs(pname):
            if ename not in encoder_names:
                errs.append(
                    f"predictor {pname!r} reads encoder {ename!r}; "
                    f"model has {sorted(encoder_names)}"
                )


def _validate_losses(pipe: Pipeline, slot_names, errs: list[str]) -> None:
    """Loss contracts (shared by both modes): slots, heads, text."""
    for loss_fn in pipe.losses:
        for s in loss_fn.requires_slots:
            if s not in slot_names:
                errs.append(
                    f"loss {loss_fn.name!r} requires slot {s!r}; "
                    f"batch provides {sorted(slot_names)}"
                )
        for h in loss_fn.requires_heads:
            if h not in pipe.model.head_names():
                errs.append(
                    f"loss {loss_fn.name!r} requires head {h!r}; "
                    f"model has {pipe.model.head_names()}"
                )
        encoder_names = set(pipe.model.encoders.keys())
        frozen = pipe.model.frozen_names()
        for e in loss_fn.requires_encoders:
            if e not in encoder_names:
                errs.append(
                    f"loss {loss_fn.name!r} requires encoder {e!r}; "
                    f"model has {sorted(encoder_names)}"
                )
        for e in loss_fn.requires_frozen_encoders:
            if e not in encoder_names:
                errs.append(
                    f"loss {loss_fn.name!r} requires encoder {e!r}; "
                    f"model has {sorted(encoder_names)}"
                )
            elif e not in frozen:
                errs.append(
                    f"loss {loss_fn.name!r} needs encoder {e!r} declared "
                    "frozen: true -- it updates it outside the optimizer "
                    "(an EMA target), so the optimizer must not train it"
                )
        for p in loss_fn.requires_predictors:
            if p not in pipe.model.predictor_names():
                errs.append(
                    f"loss {loss_fn.name!r} requires predictor {p!r}; "
                    f"model has {pipe.model.predictor_names()}"
                )
        if loss_fn.requires_text and K.CAP_TEXT_REPRESENTATION not in pipe.model.capabilities():
            errs.append(f"loss {loss_fn.name!r} needs a text encoder; model has none")


def _validate_offline(pipe: Pipeline, driver: OfflineDriver, errs: list[str]) -> None:
    """Offline (IID) mode: views must exist; batches carry no text/labels."""
    need = "video" if driver.unit == "clip" else "image"
    if need not in pipe.model.modalities().values():
        errs.append(f"offline batches are {driver.unit}s, but no encoder has modality {need!r}")
    for v in driver.views:
        if v not in pipe.views.names():
            errs.append(f"driver.batch: unknown view {v!r} (declared: {pipe.views.names()})")
    for loss_fn in pipe.losses:
        if loss_fn.requires_text:
            errs.append(
                f"loss {loss_fn.name!r} needs text, but offline v1 batches are "
                "frames-only (unit=pair is a planned extension)"
            )
