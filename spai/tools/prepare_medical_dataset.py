# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Arranges two independent class directories (authentic / generated) into the
`0_real` / `1_fake` train-val-test layout that `spai.tools.create_dir_csv` and the
phase-two training pipeline expect.

Also equalizes acquisition properties (resolution, file format, colour mode) between
the two classes. SPAI is meant to key on spectral inconsistencies a generator
introduces; if the two classes also differ in how they were captured/encoded, a
classifier can separate them on that shortcut alone, so `--audit-only` reports it and
`--target-size` / `--recode` / `--to-gray` remove it.
"""

import collections
import random
import re
from pathlib import Path
from typing import Optional

import click
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# Any single filename-derived group covering more than this share of a class's images
# is treated as a broken grouping (e.g. a shared literal prefix such as "fake_" rather
# than a per-subject id) and per-image grouping is used for that class instead.
_MAX_GROUP_SHARE = 0.05


def find_images(root: Path) -> list[Path]:
    """Returns every image file under `root`, sorted for reproducible sampling."""
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def audit_images(paths: list[Path], sample_size: int) -> dict[str, collections.Counter]:
    """Reports the resolution / format / colour-mode distribution of a sample of images."""
    sample = paths if len(paths) <= sample_size else random.sample(paths, sample_size)
    sizes: collections.Counter = collections.Counter()
    formats: collections.Counter = collections.Counter()
    modes: collections.Counter = collections.Counter()
    for path in sample:
        try:
            with Image.open(path) as img:
                sizes[img.size] += 1
                formats[img.format or path.suffix.lstrip(".").upper()] += 1
                modes[img.mode] += 1
        except Exception:
            continue
    return {"sizes": sizes, "formats": formats, "modes": modes}


def _dominant_share(counter: collections.Counter) -> tuple[Optional[object], float]:
    total = sum(counter.values())
    if not total:
        return None, 0.0
    value, count = counter.most_common(1)[0]
    return value, count / total


def report_parity(
    real_audit: dict[str, collections.Counter],
    fake_audit: dict[str, collections.Counter],
    dominance_threshold: float = 0.6,
) -> list[str]:
    """Flags fields where both classes are dominated by a different value.

    Only fires when each class is itself dominated by one value (>= `dominance_threshold`)
    and those values differ, i.e. a systematic difference a classifier could exploit as a
    shortcut - not incidental variation already present within a single class.
    """
    warnings: list[str] = []
    for field, label in (
        ("sizes", "resolution"), ("formats", "file format"), ("modes", "colour mode")
    ):
        real_value, real_share = _dominant_share(real_audit[field])
        fake_value, fake_share = _dominant_share(fake_audit[field])
        if real_value is None or fake_value is None:
            continue
        if (real_value != fake_value
                and real_share >= dominance_threshold
                and fake_share >= dominance_threshold):
            warnings.append(
                f"Dominant {label} differs between classes: authentic is mostly "
                f"{real_value} ({real_share:.0%} of sample), generated is mostly "
                f"{fake_value} ({fake_share:.0%} of sample)."
            )
    return warnings


def _group_id(path: Path, pattern: Optional[re.Pattern]) -> str:
    if pattern is not None:
        match = pattern.match(path.name)
        if match and match.groupdict().get("group"):
            return match.group("group")
    return path.stem


def _effective_groups(
    images: list[Path], pattern: Optional[re.Pattern], class_label: str
) -> list[str]:
    """Groups images by `_group_id`, guarding against a pathological grouping.

    A regex meant to capture a per-subject id (e.g. NIH's `<patient>_<followup>.png`)
    can accidentally match a literal prefix shared by an entire class (e.g. a
    "synthetic_" filename convention), which would dump that whole class into a single
    train/val/test split. Falling back to one group per image is always safe.
    """
    groups = [_group_id(p, pattern) for p in images]
    if pattern is not None and groups:
        largest_share = collections.Counter(groups).most_common(1)[0][1] / len(groups)
        if largest_share > _MAX_GROUP_SHARE:
            print(
                f"[prepare_medical_dataset] --group-regex produced a group covering "
                f"{largest_share:.0%} of {class_label} images - that looks like a shared "
                f"filename prefix rather than a per-subject id. Falling back to one group "
                f"per image for this class."
            )
            groups = [p.stem for p in images]
    return groups


def _split_groups(
    groups: list[str], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, str]:
    unique_groups = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    n = len(unique_groups)
    n_train = min(round(n * train_ratio), n)
    n_val = min(round(n * val_ratio), n - n_train)

    assignment: dict[str, str] = {}
    for group in unique_groups[:n_train]:
        assignment[group] = "train"
    for group in unique_groups[n_train:n_train + n_val]:
        assignment[group] = "val"
    for group in unique_groups[n_train + n_val:]:
        assignment[group] = "test"
    return assignment


def _format_for_ext(ext: str) -> tuple[str, str]:
    """Maps a (possibly missing/unsupported) extension to a Pillow format, defaulting to PNG."""
    mapping = {
        "png": "PNG", "jpg": "JPEG", "jpeg": "JPEG",
        "bmp": "BMP", "tif": "TIFF", "tiff": "TIFF", "webp": "WEBP",
    }
    fmt = mapping.get(ext.lower())
    return (ext.lower(), fmt) if fmt else ("png", "PNG")


def _prepare_image(
    img: Image.Image, target_size: Optional[int], resize_mode: str, to_gray: bool
) -> Image.Image:
    img = img.convert("L") if to_gray else img.convert("RGB")
    if target_size is None:
        return img
    if resize_mode == "resize":
        return img.resize((target_size, target_size), Image.BICUBIC)
    if resize_mode != "crop":
        raise click.BadParameter(f"Unsupported resize mode: {resize_mode}")
    # Center-crop images already large enough in both dimensions, leaving their spectrum
    # untouched. Resampling is a low-pass filter that attenuates exactly the high
    # frequencies SPAI depends on, so it is used only as a fallback for undersized images.
    width, height = img.size
    if width < target_size or height < target_size:
        scale = target_size / min(width, height)
        img = img.resize(
            (max(target_size, round(width * scale)), max(target_size, round(height * scale))),
            Image.BICUBIC,
        )
        width, height = img.size
    left, top = (width - target_size) // 2, (height - target_size) // 2
    return img.crop((left, top, left + target_size, top + target_size))


def _arrange_class(
    images: list[Path],
    groups: list[str],
    class_dir_name: str,
    class_tag: str,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    target_size: Optional[int],
    resize_mode: str,
    recode: Optional[str],
    to_gray: bool,
) -> collections.Counter:
    assignment = _split_groups(groups, train_ratio, val_ratio, seed)
    needs_reencode = target_size is not None or recode is not None or to_gray
    counts: collections.Counter = collections.Counter()

    for i, (path, group) in enumerate(zip(images, groups)):
        split = assignment[group]
        dest_dir = out_dir / split / class_dir_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        ext, save_format = _format_for_ext(recode or path.suffix.lstrip("."))
        dest_path = dest_dir / f"{class_tag}_{i:06d}.{ext}"

        if needs_reencode:
            with Image.open(path) as img:
                prepared = _prepare_image(img, target_size, resize_mode, to_gray)
                prepared.save(dest_path, format=save_format)
        else:
            dest_path.write_bytes(path.read_bytes())
        counts[split] += 1

    return counts


@click.command()
@click.option("--real-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Directory (searched recursively) of authentic images -> class 0.")
@click.option("--fake-dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Directory (searched recursively) of generated images -> class 1.")
@click.option("-o", "--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--train-ratio", type=float, default=0.8, show_default=True)
@click.option("--val-ratio", type=float, default=0.1, show_default=True)
@click.option("--test-ratio", type=float, default=0.1, show_default=True)
@click.option("--group-regex", type=str, default=None,
              help="Regex with a named 'group' capture, matched against each filename, so that "
                   "e.g. all images of one patient land in a single split. Falls back to one "
                   "group per image when it does not match, or when it matches too broadly.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--max-per-class", type=int, default=None,
              help="Cap each class to this many images (randomly sampled) before splitting.")
@click.option("--target-size", type=int, default=None,
              help="Bring both classes to this square size. Omit to leave images as-is.")
@click.option("--resize-mode", type=click.Choice(["crop", "resize"]), default="crop", show_default=True)
@click.option("--recode", type=click.Choice(["png", "jpeg"]), default=None,
              help="Re-encode both classes into one file format. Omit to keep source formats.")
@click.option("--to-gray", is_flag=True, default=False, help="Discard chrominance.")
@click.option("--audit-sample", type=int, default=400, show_default=True)
@click.option("--audit-only", is_flag=True, default=False,
              help="Only print the acquisition-parity audit of the source directories; build "
                   "nothing. --output-dir is still required but unused.")
def main(
    real_dir: Path,
    fake_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    group_regex: Optional[str],
    seed: int,
    max_per_class: Optional[int],
    target_size: Optional[int],
    resize_mode: str,
    recode: Optional[str],
    to_gray: bool,
    audit_sample: int,
    audit_only: bool,
) -> None:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise click.BadParameter(f"--train-ratio/--val-ratio/--test-ratio must sum to 1.0, "
                                  f"got {ratio_sum}")

    real_images = find_images(Path(real_dir))
    fake_images = find_images(Path(fake_dir))
    print(f"Found {len(real_images)} authentic images under {real_dir}")
    print(f"Found {len(fake_images)} generated images under {fake_dir}")
    if not real_images or not fake_images:
        raise click.ClickException("No images found under --real-dir and/or --fake-dir.")

    if audit_only:
        print("\n=== Acquisition audit (source images, before any preparation) ===")
        real_audit = audit_images(real_images, audit_sample)
        fake_audit = audit_images(fake_images, audit_sample)
        for label, audit in (("authentic", real_audit), ("generated", fake_audit)):
            print(f"\n{label}:")
            for field in ("sizes", "formats", "modes"):
                top = ", ".join(f"{k}: {v}" for k, v in audit[field].most_common(5))
                print(f"  {field:<9}: {top}")
        warnings = report_parity(real_audit, fake_audit)
        if warnings:
            print("\nPotential shortcuts (address with --target-size / --recode / --to-gray):")
            for warning in warnings:
                print(f"  [WARNING] {warning}")
        else:
            print("\nNo dominant acquisition difference detected in this sample.")
        return

    pattern = re.compile(group_regex) if group_regex else None

    if max_per_class is not None:
        if len(real_images) > max_per_class:
            real_images = random.Random(seed).sample(real_images, max_per_class)
        if len(fake_images) > max_per_class:
            fake_images = random.Random(seed + 1).sample(fake_images, max_per_class)

    real_groups = _effective_groups(real_images, pattern, "authentic")
    fake_groups = _effective_groups(fake_images, pattern, "generated")

    output_dir = Path(output_dir)
    real_counts = _arrange_class(
        real_images, real_groups, "0_real", "real", output_dir,
        train_ratio, val_ratio, seed, target_size, resize_mode, recode, to_gray,
    )
    fake_counts = _arrange_class(
        fake_images, fake_groups, "1_fake", "fake", output_dir,
        train_ratio, val_ratio, seed + 1, target_size, resize_mode, recode, to_gray,
    )

    print("\nImages written per split:")
    for split in ("train", "val", "test"):
        print(f"  {split:<5} authentic: {real_counts[split]:>6}  generated: {fake_counts[split]:>6}")


if __name__ == "__main__":
    main()
