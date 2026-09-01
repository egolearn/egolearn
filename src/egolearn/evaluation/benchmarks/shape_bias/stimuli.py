"""Deterministic stimuli for Tartaglini et al. (2022), Experiment 3.

The upstream repository does not include a license. Its assets are therefore
downloaded from a pinned commit and checksum-verified, never redistributed by
EgoLearn.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SOURCE_REPOSITORY = "https://github.com/alexatartaglini/developmental-shape-bias"
SOURCE_COMMIT = "09277d997e5d2924b83c5b6d1d792475a200a0b4"
CANVAS_SIZE = 224
OBJECT_SIZE = 180
PERCENT_ORIGINAL_SIZE = 80
DEFAULT_SEED = 1

ASSET_SHA256 = {
    "novel-masks/dax.png": "499b6fedb338d97598726721ab190c4f4ac5b5e46918f3fe08584dbbbf1a3dcd",
    "novel-masks/gam.png": "fe2934c8d7f76cb13fa026cb34a0f9b83b9564800892e8f3d5a7184b5162b09c",
    "novel-masks/gub.png": "1a41489d67dc69665eb25f4c787b0f4a6a1d7276508ce0532ad07d7f3e1f74f6",
    "novel-masks/kag.png": "5e26f822616a2413ed502d01c9d481e65f92f2b5a97ee4759ebaee0068f140bc",
    "novel-masks/kev.png": "0ab5d823efe92a98ef0d8ebae273a8cbd46e094bcb90c83cd4f92ed4893d6699",
    "novel-masks/laz.png": "b849f932f08468c22b0b8648852485aa0d8b77ae400687ae1fff4a6c66821d99",
    "novel-masks/lim.png": "0c503462836ca81721ca901bee532124aa4832e936aba5b3c285d8e27d5e1dbc",
    "novel-masks/lok.png": "33f144fcd9a2325fea93fbae4b4950119270b4a75a51f877dcb16b1d4880b8b9",
    "novel-masks/lug.png": "8c49f3626bd1cec30a64438a3901f96137881de004e7e63908ca182fe9c1b474",
    "novel-masks/ruc.png": "36156e15f0490829b539ab97fd486f16a0083cf3ab3239f9a83f628e2c15ddef",
    "novel-masks/sut.png": "2af3ca64c0899b3a96a78393041ff03e4473f7721b7ab7c3324ccd6aae843fc2",
    "novel-masks/tof.png": "e49fe0e94fdc785e32c81798c943295bba5ef2344484f32973003fae8bfb904a",
    "novel-masks/wif.png": "546fe46e61b4c112a7180398af87d201411db76cb2d8b3562f84bd4aea324ffd",
    "novel-masks/wob.png": "34f1bdc220bf711bd721ae3b38e4cc4a8ae10459ddeef353c3f1f193967359c8",
    "novel-masks/zot.png": "0ab22be80682b7c55b01a93f253024b7d56a40b136e66a721a57845f38f24a25",
    "novel-masks/zup.png": "d873b7118324c840e2872acc33a39c8f3349b6ceb0461d84af72312d961a864d",
    "brodatz-textures/D101.png": "9d4aad8d658e1a17744bf35fdd8dfa77de0deebf2ba0eaca979205351b2578cb",
    "brodatz-textures/D104.png": "836377b6adbc4be670b45ef8ae9a5d8ac6231a41dd37e0dec309d01dd29e18d1",
    "brodatz-textures/D108.png": "90de9e00a9a4d09b6b26cd99d58f7332e418a213b17cf67fb418b2ddc40ad816",
    "brodatz-textures/D111.png": "2e25ad5d1479f1c4747c769dc8d32f7b4bbcf8d3d6dbe745de3a4c7c4c433040",
    "brodatz-textures/D4.png": "f70e846ee9e8a01c3f2fe06aeaed02e1adeb3d1e68cd312d991cad32ec8f0dc7",
    "brodatz-textures/D47.png": "fc02cd1145a08c484501c5218ddc2f55a6ef5a3cd95154f951e974a7e972727a",
    "brodatz-textures/D51.png": "4afa21d3179e165147e405273c6ad89a79e2a0f084f15cfc2255152818d29e69",
    "brodatz-textures/D56.png": "72bf0e3d2ea5b2f8a4959d1480024004435062a882b5498109897f8ab0115370",
    "brodatz-textures/D62.png": "f09fa3f0e8b2e68d72ae65404c6a2de51b91ecb461fc743f5849f6c3b57f6283",
    "brodatz-textures/D67.png": "1418463bf0ffa9dee699482cc9d2efe0276e9231b491ce4a5727429bc116c4dd",
    "brodatz-textures/D74.png": "757bbf4ee991327e895ca7a477d27f4e3a1ce80af2aa2d2fa1cd872c81398cf0",
    "brodatz-textures/D83.png": "5895671d9858bee7804156e10cecc086d8a140359178781dc1f79bcd64bd959f",
    "brodatz-textures/D87.png": "5b0c416f0c72a4f316207f14a532bf2f4ffa96004535ab9cdc5db258cfcddccd",
    "brodatz-textures/D88.png": "309a47b1f82f85063e3e6bed55ed26e78891388ea34fd14010935d9bc3fd054c",
    "brodatz-textures/D93.png": "c9967992c8bd31c1b3c99ed167ae29a029817cbd3a34c265ef14d094141d7e16",
    "brodatz-textures/D95.png": "3dd2acd929e8416f43069714e3cded38cc1e3a6e2b815aa0b70e7af985e78588",
}


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_shape_bias_assets(
    asset_dir: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Download and verify the 16 novel masks and 16 Brodatz textures."""
    root = Path(asset_dir)
    base_url = (
        "https://raw.githubusercontent.com/alexatartaglini/"
        f"developmental-shape-bias/{SOURCE_COMMIT}/stimuli"
    )
    records: list[dict[str, str]] = []
    for relative_path, expected_sha256 in ASSET_SHA256.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256_file(target) == expected_sha256:
            records.append({"path": relative_path, "sha256": expected_sha256, "status": "verified"})
            continue
        if target.exists() and not force:
            raise ValueError(f"checksum mismatch for {target}; pass force=True to replace it")

        temporary = target.with_suffix(target.suffix + ".download")
        try:
            with (
                urllib.request.urlopen(f"{base_url}/{relative_path}") as response,
                temporary.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
            actual_sha256 = sha256_file(temporary)
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"checksum mismatch for {relative_path}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        records.append({"path": relative_path, "sha256": expected_sha256, "status": "downloaded"})

    return {"repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT, "assets": records}


def _source_paths(asset_dir: Path, subdirectory: str) -> list[Path]:
    paths = sorted((asset_dir / subdirectory).glob("*.png"))
    if len(paths) != 16:
        raise ValueError(f"expected 16 files in {asset_dir / subdirectory}, found {len(paths)}")
    for path in paths:
        relative_path = f"{subdirectory}/{path.name}"
        expected = ASSET_SHA256.get(relative_path)
        if expected is None or sha256_file(path) != expected:
            raise ValueError(f"source asset failed checksum validation: {path}")
    return paths


def _render_stimulus(
    mask_path: Path,
    texture_path: Path,
    rng: random.Random,
) -> tuple[Image.Image, dict[str, int]]:
    with Image.open(mask_path) as source_mask:
        mask = source_mask.convert("RGBA")
    if mask.size != (CANVAS_SIZE, CANVAS_SIZE):
        raise ValueError(f"expected a 224x224 mask, got {mask.size} for {mask_path}")

    with Image.open(texture_path) as source_texture:
        texture = source_texture.convert("RGBA").resize(
            (CANVAS_SIZE * 2, CANVAS_SIZE * 2), Image.Resampling.BICUBIC
        )
    crop_bound = texture.width - mask.width
    crop_x = rng.randint(0, crop_bound)
    crop_y = rng.randint(0, crop_bound)
    texture = texture.crop((crop_x, crop_y, crop_x + CANVAS_SIZE, crop_y + CANVAS_SIZE))

    mask_array = np.asarray(mask)
    foreground = np.all(mask_array[:, :, :3] == 0, axis=2)
    alpha = np.where(foreground, mask_array[:, :, 3], 0).astype(np.uint8)
    alpha_image = Image.fromarray(alpha)
    textured_shape = Image.new("RGBA", mask.size, (255, 255, 255, 0))
    textured_shape.paste(texture, (0, 0), mask=alpha_image)
    textured_shape = textured_shape.resize((OBJECT_SIZE, OBJECT_SIZE), Image.Resampling.NEAREST)

    placement_bound = CANVAS_SIZE - OBJECT_SIZE
    offset_x = rng.randint(0, placement_bound)
    offset_y = rng.randint(0, placement_bound)
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    canvas.paste(
        textured_shape.convert("RGB"),
        (offset_x, offset_y),
        mask=textured_shape.getchannel("A"),
    )
    return canvas, {
        "texture_crop_x": crop_x,
        "texture_crop_y": crop_y,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


def prepare_shape_bias_stimuli(
    asset_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Render the deterministic 80%-size, unaligned novel-stimulus grid."""
    assets = Path(asset_dir)
    output = Path(output_dir)
    masks = _source_paths(assets, "novel-masks")
    textures = _source_paths(assets, "brodatz-textures")
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for mask_path in masks:
        shape = mask_path.stem
        for texture_path in textures:
            texture = texture_path.stem
            image, placement = _render_stimulus(mask_path, texture_path, rng)
            relative_path = Path("stimuli") / shape / f"{shape}-{texture}.png"
            target = output / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            records.append(
                {
                    "shape": shape,
                    "texture": texture,
                    "path": relative_path.as_posix(),
                    "sha256": sha256_file(target),
                    **placement,
                }
            )

    manifest = {
        "protocol": "tartaglini_exp3_novel_80pct_unaligned",
        "source": {
            "paper": "https://arxiv.org/abs/2202.08340",
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "asset_sha256": ASSET_SHA256,
        },
        "rendering": {
            "seed": seed,
            "canvas_size": CANVAS_SIZE,
            "object_size": OBJECT_SIZE,
            "percent_original_size": PERCENT_ORIGINAL_SIZE,
            "placement": "unaligned",
            "texture_resize": [CANVAS_SIZE * 2, CANVAS_SIZE * 2],
            "texture_resampling": "bicubic",
            "object_resampling": "nearest",
            "background_rgb": [255, 255, 255],
        },
        "shapes": [path.stem for path in masks],
        "textures": [path.stem for path in textures],
        "n_stimuli": len(records),
        "n_triplets": len(records) * (len(masks) - 1) * (len(textures) - 1),
        "stimuli": records,
    }
    if manifest["n_stimuli"] != 256 or manifest["n_triplets"] != 57_600:
        raise AssertionError("the shape-bias protocol has invalid cardinality")
    write_json(output / "stimulus_manifest.json", manifest)
    return manifest


def load_stimulus_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"expected a mapping in {path}")
    return manifest


def ordered_stimulus_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_pair = {(record["shape"], record["texture"]): record for record in manifest["stimuli"]}
    return [
        by_pair[(shape, texture)]
        for shape in manifest["shapes"]
        for texture in manifest["textures"]
    ]


def validate_stimulus_manifest(
    manifest: dict[str, Any],
    root: str | Path,
    *,
    verify_hashes: bool = True,
) -> None:
    shapes = manifest.get("shapes", [])
    textures = manifest.get("textures", [])
    records = manifest.get("stimuli", [])
    expected_pairs = {(shape, texture) for shape in shapes for texture in textures}
    actual_pairs = {(record["shape"], record["texture"]) for record in records}
    if len(set(shapes)) != 16 or len(set(textures)) != 16 or len(records) != 256:
        raise ValueError("expected 16 unique shapes, 16 unique textures, and 256 stimuli")
    if actual_pairs != expected_pairs or len(actual_pairs) != len(records):
        raise ValueError("stimulus manifest is not a complete shape-by-texture grid")
    if manifest.get("n_triplets") != 57_600:
        raise ValueError("expected 57,600 shape-versus-texture triplets")

    root_path = Path(root)
    for record in records:
        path = root_path / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.size != (CANVAS_SIZE, CANVAS_SIZE) or image.mode != "RGB":
                raise ValueError(f"expected a 224x224 RGB image: {path}")
        if verify_hashes and sha256_file(path) != record["sha256"]:
            raise ValueError(f"stimulus checksum mismatch: {path}")
