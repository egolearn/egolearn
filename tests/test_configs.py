from hydra import compose, initialize_config_module


def test_root_config_composes_from_installed_package() -> None:
    # pkg:// composition is the guarantee that entry points never require
    # running from the repo root (unlike the source repos this consolidates).
    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = compose(config_name="config")
    assert cfg.seed == 0
    assert cfg.wandb.entity == "nyu-mengye-group"
    assert cfg.wandb.mode == "disabled"


def test_toy_streaming_config_composes_and_builds() -> None:
    # the packaged toy config exercises the whole hydra -> registry path:
    # defaults composition, kind:name instantiation, build-time validation
    from omegaconf import OmegaConf

    import egolearn

    with initialize_config_module(config_module="egolearn.configs", version_base="1.3"):
        cfg = compose(config_name="toy_streaming")
    assert cfg.seed == 7  # _self_ wins over the root default
    plain = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(plain, dict)
    plain.pop("dry_run"), plain.pop("resume")
    pipe = egolearn.build(plain)
    assert pipe.mode == "streaming"
    assert pipe.model.head_names() == ["cls", "proj"]
