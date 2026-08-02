from hydra import compose, initialize_config_module


def test_root_config_composes_from_installed_package() -> None:
    # pkg:// composition is the guarantee that entry points never require
    # running from the repo root (unlike the source repos this consolidates).
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = compose(config_name="config")
    assert cfg.seed == 0
    assert cfg.wandb.entity == "nyu-mengye-group"
    assert cfg.wandb.mode == "disabled"
