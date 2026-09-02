"""Download and render the pinned shape-bias benchmark stimuli."""

from __future__ import annotations

import argparse

from egolearn.evaluation.benchmarks.shape_bias.stimuli import (
    DEFAULT_SEED,
    download_shape_bias_assets,
    prepare_shape_bias_stimuli,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Tartaglini Experiment 3 at 80% size, unaligned."
    )
    parser.add_argument("--asset-dir", default="data/shape_bias/source")
    parser.add_argument("--output-dir", default="data/shape_bias/80_unaligned")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    source = download_shape_bias_assets(args.asset_dir, force=args.force_download)
    manifest = prepare_shape_bias_stimuli(args.asset_dir, args.output_dir, seed=args.seed)
    downloaded = sum(asset["status"] == "downloaded" for asset in source["assets"])
    print(
        f"Prepared {manifest['n_stimuli']} stimuli and {manifest['n_triplets']:,} "
        f"triplets in {args.output_dir} ({downloaded} assets downloaded)."
    )


if __name__ == "__main__":
    main()
