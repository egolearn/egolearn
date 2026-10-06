"""Shared synthetic training configurations."""

import pytest


@pytest.fixture
def toy_config():
    def build(tmp_path, dim: int = 64):
        return {
            "run_dir": str(tmp_path / "run"),
            "seed": 7,
            "device": "cpu",
            "stream": {
                "_target_": "stream:toy",
                "n_sessions": 1,
                "blocks_per_session": 4,
                "block_len": 40,
                "img_size": 64,
            },
            "views": {
                "v1": {"name": "simclr_strong", "size": 64, "view_idx": 0},
                "v2": {"name": "simclr_strong", "size": 64, "view_idx": 1},
            },
            "label_stores": ["fixed"],
            "admission": [
                {"_target_": "admission:frame_emitter", "subsample": 1},
                {
                    "_target_": "admission:fixed_window_labeler",
                    "class_length": 40,
                    "label_store": "fixed",
                },
            ],
            "memories": {
                "frames": {
                    "tiers": {
                        "short": {"capacity": 30, "update": "fifo"},
                        "long": {"capacity": 60, "update": "reservoir"},
                    },
                    "admit_to": ["short", "long"],
                    "overflow": {},
                }
            },
            "routes": {"RawFrame": "frames"},
            "recipe": {
                "slots": [
                    {
                        "name": "curr",
                        "source": {"kind": "tier", "memory": "frames", "tier": "short"},
                        "n": 4,
                        "unit": "frame",
                        "views": ["v1", "v2"],
                        "labels_from": "fixed",
                        "strict": True,
                    },
                ]
            },
            "model": {
                "encoders": {"vision": {"_target_": "encoder:toy_cnn", "dim": dim}},
                "heads": {
                    "proj": {"encoder": "vision", "_target_": "head:mlp", "sizes": [dim, dim, 32]},
                    "cls": {"encoder": "vision", "_target_": "head:mlp", "sizes": [dim, dim, 8]},
                },
            },
            "losses": [
                {
                    "_target_": "loss:ntxent",
                    "name": "inst",
                    "slots": ["curr"],
                    "head": "proj",
                    "temperature": 0.1,
                    "weight": 1.0,
                },
                {
                    "_target_": "loss:tcl",
                    "name": "temp",
                    "slots": ["curr"],
                    "head": "cls",
                    "num_classes": 8,
                    "weight": 1.0,
                },
            ],
            "solver": {
                "opt": "sgd",
                "groups": [{"match": ".*", "lr": 0.01, "weight_decay": 0.0}],
                "schedule": {"constant": True},
            },
            "driver": {
                "advance": {"kind": "observations", "n": 8},
                "rate": {"steps_per_admit": 1, "resample": True},
                "warmup": {"admitted": 20},
                "max_steps": 2,
            },
            "hooks": {"log_every": 1},
        }

    return build
