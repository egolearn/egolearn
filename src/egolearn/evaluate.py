"""Standalone checkpoint evaluation using Hydra configuration.

    python -m egolearn.evaluate --config-name imagenet_linear_eval \
        checkpoint=run/final.pth data_root=/datasets/imagenet
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from egolearn.run_record import record_run


@hydra.main(version_base="1.3", config_path="pkg://egolearn.configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    from egolearn.evaluation import EvaluationError, evaluate_checkpoint

    plain = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    assert isinstance(plain, dict)
    with record_run(plain, "eval") as record:
        plain = record.config if record is not None else plain
        try:
            result = evaluate_checkpoint(
                plain["checkpoint"],
                model_config=plain.get("model_config"),
                benchmarks=plain["benchmarks"],
                device=plain.get("device", "cpu"),
                seed=plain.get("seed", 0),
                imports=plain.get("_imports_", []),
                trusted_snapshot=plain.get("trusted_snapshot", False),
            )
        except EvaluationError as exc:
            if plain.get("output_json"):
                exc.partial_result.save_json(plain["output_json"])
            raise
        if record is not None:
            record.evaluation(result)
        if plain.get("output_json"):
            result.save_json(plain["output_json"])
        print(json.dumps(result.to_dict(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
