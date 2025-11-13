#!/usr/bin/env python3
"""
Utility to build composite microrobot datasets driven by a YAML config.

Features:
- Merge multiple microrobot datasets into a new YOLO-style dataset folder.
- Optionally generate augmented variants per image using ultrasound-aware transforms.
- Supports copy or symlink modes and collision-safe naming.
- Emits a dataset YAML manifest compatible with YOLO/TPH-YOLOv5 training CLIs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

try:
    import albumentations as A
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "albumentations is required for the dataset builder. Install it via the project "
        "environment files before running this script."
    ) from exc
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_models.augmentation_config import (  # noqa: E402
    AugmentationPipelineSpec,
    BuilderConfig,
    build_pipeline,
    load_builder_config,
)


DATA_ROOT = ROOT / "data"


# --------------------------- Core builder --------------------------------------


@dataclass
class Sample:
    microrobot: str
    source_split: str
    image_path: Path
    label_path: Path
    target_split: str


class MultirobotDatasetBuilder:
    def __init__(self, config: BuilderConfig, dry_run: bool = False, verbose: bool = True) -> None:
        self.config = config
        self.dry_run = dry_run
        self.verbose = verbose
        self.output_root = Path(config.dataset.output_root).resolve()
        self.dataset_dir = self.output_root / config.dataset.name
        self.images_dirs = {split: self.dataset_dir / "images" / split for split in config.dataset.splits}
        self.labels_dirs = {split: self.dataset_dir / "labels" / split for split in config.dataset.splits}
        self.naming = config.dataset.naming
        self.generated_names: Dict[str, int] = {}
        self.split_ratios = config.dataset.split_ratios
        self.shuffle_seed = config.dataset.shuffle_seed
        self.shuffle_samples = config.dataset.shuffle_samples
        self.total_ratio = sum(self.split_ratios.values()) if self.split_ratios else None
        self.stats = {
            "original": 0,
            "augmented": 0,
            "skipped_missing_label": 0,
            "skipped_empty_boxes": 0,
            "assigned_split_counts": {split: 0 for split in config.dataset.splits},
            "configured_split_ratios": self.split_ratios or {},
        }
        self.pipelines: List[Tuple[AugmentationPipelineSpec, A.Compose, bool]] = []
        for spec in config.augmentations:
            built = build_pipeline(spec)
            if built is None:
                continue
            pipeline, modifies = built
            self.pipelines.append((spec, pipeline, modifies))

    # --------------------- public API -----------------------------------------

    def run(self) -> None:
        samples = self._collect_samples()
        if self.split_ratios:
            self._assign_splits_by_ratio(samples)
        self._update_assigned_counts(samples)
        if self.verbose:
            self._log(f"Collected {len(samples)} samples across sources.")
        if not self.dry_run:
            self._prepare_output_dirs()
        iterator = tqdm(
            samples,
            desc="Building dataset",
            disable=not self.verbose,
            unit="sample",
        )
        for sample in iterator:
            self._process_sample(sample)
        if not self.dry_run:
            self._write_manifest()
        self._log(json.dumps(self.stats, indent=2))

    # --------------------- internal helpers -----------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _collect_samples(self) -> List[Sample]:
        samples: List[Sample] = []
        for source in self.config.sources:
            for split in source.splits:
                images_dir = source.images_dir or (DATA_ROOT / source.microrobot / "images" / split)
                labels_dir = source.labels_dir or (DATA_ROOT / source.microrobot / "labels" / split)
                if not images_dir.exists():
                    raise FileNotFoundError(f"Images dir missing: {images_dir}")
                if not labels_dir.exists():
                    raise FileNotFoundError(f"Labels dir missing: {labels_dir}")
                for image_path in sorted(images_dir.glob("*")):
                    if image_path.is_dir():
                        continue
                    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                        continue
                    label_path = labels_dir / f"{image_path.stem}.txt"
                    samples.append(Sample(source.microrobot, split, image_path, label_path, split))
        return samples

    def _assign_splits_by_ratio(self, samples: List[Sample]) -> None:
        if not samples or not self.split_ratios:
            return
        if self.shuffle_samples:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(samples)
        else:
            samples.sort(key=self._sample_sort_key)

        grouped: Dict[str, List[Sample]] = {}
        for sample in samples:
            grouped.setdefault(sample.microrobot, []).append(sample)
        for subset in grouped.values():
            self._assign_subset_by_ratio(subset)

    @staticmethod
    def _sample_sort_key(sample: Sample) -> tuple:
        return (
            sample.image_path.name.lower(),
            sample.microrobot.lower(),
            sample.source_split.lower(),
            str(sample.image_path.parent).lower(),
        )

    def _assign_subset_by_ratio(self, subset: List[Sample]) -> None:
        if not subset:
            return
        total = len(subset)
        total_ratio = self.total_ratio or 0.0
        if total_ratio <= 0.0:
            raise ValueError("split ratios must sum to a positive value.")
        splits = self.config.dataset.splits
        allocations: Dict[str, int] = {}
        remaining = total
        for idx, split in enumerate(splits):
            ratio = self.split_ratios.get(split, 0.0)
            if idx == len(splits) - 1:
                count = remaining
            else:
                count = int(round((ratio / total_ratio) * total)) if total_ratio > 0 else 0
                count = max(0, min(remaining, count))
                remaining -= count
            allocations[split] = count
        if remaining != 0:
            allocations[splits[-1]] = allocations.get(splits[-1], 0) + remaining

        cursor = 0
        for split in splits:
            count = allocations.get(split, 0)
            if count <= 0:
                continue
            for _ in range(count):
                if cursor >= len(subset):
                    break
                subset[cursor].target_split = split
                cursor += 1
        while cursor < len(subset):
            subset[cursor].target_split = splits[-1]
            cursor += 1

    def _prepare_output_dirs(self) -> None:
        if self.dataset_dir.exists():
            if not self.config.dataset.overwrite:
                raise FileExistsError(
                    f"Target dataset directory already exists: {self.dataset_dir}. "
                    "Set dataset.overwrite=true to replace it."
                )
            shutil.rmtree(self.dataset_dir)
        for path in list(self.images_dirs.values()) + list(self.labels_dirs.values()):
            path.mkdir(parents=True, exist_ok=True)

    def _generate_base_name(self, sample: Sample) -> str:
        base = sample.image_path.stem
        prefix = sample.microrobot
        if self.naming.force_lowercase:
            prefix = prefix.lower()
            base = base.lower()
        combined = f"{prefix}{self.naming.delimiter}{base}"
        return combined

    def _reserve_name(self, name: str) -> str:
        count = self.generated_names.get(name)
        if count is None:
            self.generated_names[name] = 1
            return name
        else:
            idx = count + 1
            self.generated_names[name] = idx
            return f"{name}{self.naming.delimiter}{idx}"

    def _write_image_and_label(
        self,
        split: str,
        base_name: str,
        image: np.ndarray,
        boxes: Sequence[Tuple[float, float, float, float]],
        class_labels: Sequence[int],
        suffix: Optional[str] = None,
        image_ext: str = ".png",
    ) -> None:
        if not boxes:
            self.stats["skipped_empty_boxes"] += 1
            return
        name = base_name
        if suffix:
            name = f"{name}{self.naming.delimiter}{suffix}"
        name = self._reserve_name(name)
        images_dir = self.images_dirs[split]
        labels_dir = self.labels_dirs[split]
        image_path = images_dir / f"{name}{image_ext}"
        label_path = labels_dir / f"{name}.txt"
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if self.dry_run:
            self._log(f"[dry-run] would write image {image_path} and label {label_path}")
            return
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        cv2.imwrite(str(image_path), image)
        with label_path.open("w") as f:
            for cls, bbox in zip(class_labels, boxes):
                bbox_clamped = [
                    float(max(0.0, min(1.0, coord)))
                    for coord in bbox
                ]
                f.write(f"{cls} {' '.join(f'{c:.6f}' for c in bbox_clamped)}\n")

    def _copy_original(self, sample: Sample, base_name: str) -> None:
        if not sample.label_path.exists():
            self.stats["skipped_missing_label"] += 1
            return
        target_name = self._reserve_name(base_name)
        image_ext = sample.image_path.suffix.lower()
        image_dest = self.images_dirs[sample.target_split] / f"{target_name}{image_ext}"
        label_dest = self.labels_dirs[sample.target_split] / f"{target_name}.txt"
        if self.dry_run:
            self._log(f"[dry-run] would {self.config.dataset.copy_mode} image {sample.image_path} -> {image_dest}")
            self._log(f"[dry-run] would copy label {sample.label_path} -> {label_dest}")
            return
        if self.config.dataset.copy_mode == "copy":
            shutil.copy2(sample.image_path, image_dest)
        else:
            os.symlink(sample.image_path, image_dest)
        shutil.copy2(sample.label_path, label_dest)
        self.stats["original"] += 1

    def _process_sample(self, sample: Sample) -> None:
        base_name = self._generate_base_name(sample)
        if self.config.dataset.include_original:
            self._copy_original(sample, base_name)

        if not self.pipelines:
            return

        if sample.target_split != "train":
            return

        if not sample.label_path.exists():
            self.stats["skipped_missing_label"] += 1
            return

        image = cv2.imread(str(sample.image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            self._log(f"Warning: unable to read image {sample.image_path}, skipping.")
            return
        image_ext = sample.image_path.suffix.lower()
        with sample.label_path.open("r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            self.stats["skipped_empty_boxes"] += 1
            return

        class_labels: List[int] = []
        boxes: List[Tuple[float, float, float, float]] = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            class_labels.append(int(float(parts[0])))
            boxes.append(tuple(float(x) for x in parts[1:5]))
        if not boxes:
            self.stats["skipped_empty_boxes"] += 1
            return

        image_for_aug = image[..., None]

        for pipeline_spec, pipeline, modifies_bboxes in self.pipelines:
            for copy_idx in range(pipeline_spec.copies):
                if modifies_bboxes:
                    augmented = pipeline(image=image_for_aug, bboxes=boxes, class_labels=class_labels)
                    aug_image = augmented["image"]
                    aug_boxes = augmented["bboxes"]
                    aug_labels = augmented["class_labels"]
                else:
                    augmented = pipeline(image=image_for_aug)
                    aug_image = augmented["image"]
                    aug_boxes = boxes
                    aug_labels = class_labels
                suffix = f"{pipeline_spec.name}{copy_idx + 1 if pipeline_spec.copies > 1 else ''}"
                self._write_image_and_label(
                    split=sample.target_split,
                    base_name=base_name,
                    image=aug_image,
                    boxes=aug_boxes,
                    class_labels=aug_labels,
                    suffix=suffix,
                    image_ext=image_ext,
                )
                self.stats["augmented"] += 1

    def _write_manifest(self) -> None:
        yaml_name = self.config.dataset.yaml_filename
        if yaml_name:
            yaml_path = Path(yaml_name)
            if yaml_path.is_absolute():
                manifest_path = yaml_path
            else:
                manifest_path = self.dataset_dir / yaml_path
        else:
            manifest_path = self.dataset_dir / "dataset.yaml"
        payload: Dict[str, Any] = {
            "microrobot_type": self.config.dataset.name,
            "nc": 1,
            "names": ["microrobot"],
        }
        for split in self.config.dataset.splits:
            key = split if split in {"train", "val", "test"} else f"{split}_images"
            payload[key] = str((self.dataset_dir / "images" / split).resolve())
        manifest_path.write_text(yaml.safe_dump(payload))
        self._log(f"Wrote dataset manifest to {manifest_path}")

    def _update_assigned_counts(self, samples: List[Sample]) -> None:
        split_counts = {split: 0 for split in self.config.dataset.splits}
        for sample in samples:
            split_counts[sample.target_split] = split_counts.get(sample.target_split, 0) + 1
        self.stats["assigned_split_counts"] = split_counts


# --------------------------- CLI ------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a composite microrobot dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to the builder YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and log actions without writing files.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose logging.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    config = load_builder_config(args.config.resolve())
    builder = MultirobotDatasetBuilder(config=config, dry_run=args.dry_run, verbose=not args.quiet)
    builder.run()

if __name__ == "__main__":
    main()
