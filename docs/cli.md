# Command line

`egl train` and `egl eval` dispatch to the Hydra training and evaluation modules.
The first argument can be a packaged config name or an existing `.yaml` file;
relative file paths are resolved from the caller's working directory. `egl eval`
without a config selects the packaged `evaluate` defaults. `egl --version`
prints the installed version.

## Configuration and overrides

```bash
egl train toy_streaming seed=42 dry_run=true
egl train projects/babycl/configs/saycam_default.yaml --cfg job
egl eval path/to/suite.yaml checkpoint=runs/example/final.pth
```

Arguments after the command belong to Hydra: `--help`, `--cfg job`, `--multirun`,
`key=value`, `+key=value`, and `~key` keep their usual meaning. For example,
`egl train toy_streaming --multirun seed=1,2 dry_run=true` launches two dry runs.

Explicit files search their own directory first, then the installed config tree,
then entries supplied through `hydra.searchpath`. They can inherit packaged
defaults such as `defaults: [/config, _self_]` while overriding local components.

The shorthand expands to native Hydra config flags before any remaining arguments.
Later flags take precedence, including Hydra's accepted abbreviations:
`egl train toy_streaming --config-n=config --cfg job` selects `config`.
Native `--config-path`, `--config-name`, and `--config-dir` flags are also accepted;
supplying them later replaces the corresponding shorthand-generated setting.

The existing `python -m egolearn.train` and `python -m egolearn.evaluate` entry
points remain supported.
