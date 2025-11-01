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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
DATA_ROOT = ROOT / "data"


# --------------------------- Config dataclasses ---------------------------------


@dataclass
class TransformSpec:
    type: str
    probability: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AugmentationPipelineSpec:
    name: str
    copies: int = 1
    transforms: List[TransformSpec] = field(default_factory=list)


@dataclass
class SourceSpec:
    microrobot: str
    splits: List[str]
    images_dir: Optional[Path] = None
    labels_dir: Optional[Path] = None


@dataclass
class NamingSpec:
    strategy: str = "prefix"  # currently only prefix supported
    delimiter: str = "__"
    force_lowercase: bool = True


@dataclass
class DatasetSpec:
    name: str
    output_root: Path = DATA_ROOT
    copy_mode: str = "copy"  # copy | symlink
    include_original: bool = True
    overwrite: bool = False
    splits: List[str] = field(default_factory=lambda: ["train", "val", "test"])
    naming: NamingSpec = field(default_factory=NamingSpec)
    yaml_filename: Optional[str] = None
    split_ratios: Optional[Dict[str, float]] = None
    shuffle_seed: Optional[int] = None


@dataclass
class BuilderConfig:
    dataset: DatasetSpec
    sources: List[SourceSpec]
    augmentations: List[AugmentationPipelineSpec] = field(default_factory=list)


# --------------------------- Utilities -----------------------------------------


def _ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML at {path}: {exc}") from exc


def _resolve_optional_path(value: Optional[str]) -> Optional[Path]:
    if value in (None, "", False):
        return None
    return Path(value).expanduser().resolve()


def load_config(config_path: Path) -> BuilderConfig:
    raw = _load_yaml(config_path)

    dataset_section = raw.get("dataset")
    if not isinstance(dataset_section, dict):
        raise ValueError("Config requires a 'dataset' mapping.")

    name = dataset_section.get("name")
    if not name:
        raise ValueError("dataset.name must be provided.")
    output_root = _resolve_optional_path(dataset_section.get("output_root")) or DATA_ROOT
    copy_mode = dataset_section.get("copy_mode", "copy")
    if copy_mode not in {"copy", "symlink"}:
        raise ValueError("dataset.copy_mode must be 'copy' or 'symlink'.")
    include_original = bool(dataset_section.get("include_original", True))
    overwrite = bool(dataset_section.get("overwrite", False))
    split_ratios_section = dataset_section.get("split_ratios")
    splits_config = dataset_section.get("splits")
    split_ratios: Optional[Dict[str, float]] = None
    if split_ratios_section is not None:
        if not isinstance(split_ratios_section, dict) or not split_ratios_section:
            raise ValueError("dataset.split_ratios must be a non-empty mapping of split -> ratio.")
        split_ratios = {}
        total_ratio = 0.0
        for split_name, value in split_ratios_section.items():
            ratio = float(value)
            if ratio < 0.0:
                raise ValueError(f"dataset.split_ratios[{split_name}] must be non-negative.")
            split_ratios[split_name] = ratio
            total_ratio += ratio
        if total_ratio <= 0.0:
            raise ValueError("dataset.split_ratios must sum to a positive value.")
        splits = list(split_ratios.keys())
        if splits_config:
            user_splits = _ensure_list(splits_config)
            if user_splits != splits:
                raise ValueError("dataset.splits must match dataset.split_ratios keys when both are provided.")
    else:
        splits = _ensure_list(splits_config or ["train", "val", "test"])
    yaml_filename = dataset_section.get("yaml_filename")
    shuffle_seed_value = dataset_section.get("shuffle_seed")
    shuffle_seed = int(shuffle_seed_value) if shuffle_seed_value is not None else None

    naming_section = dataset_section.get("naming", {})
    naming = NamingSpec(
        strategy=naming_section.get("strategy", "prefix"),
        delimiter=naming_section.get("delimiter", "__"),
        force_lowercase=bool(naming_section.get("force_lowercase", True)),
    )
    if naming.strategy != "prefix":
        raise ValueError("Only naming.strategy == 'prefix' is supported currently.")

    dataset_spec = DatasetSpec(
        name=name,
        output_root=Path(output_root),
        copy_mode=copy_mode,
        include_original=include_original,
        overwrite=overwrite,
        splits=splits,
        naming=naming,
        yaml_filename=yaml_filename,
        split_ratios=split_ratios,
        shuffle_seed=shuffle_seed,
    )

    sources_section = raw.get("sources")
    if not sources_section or not isinstance(sources_section, Sequence):
        raise ValueError("Config 'sources' must be a non-empty list.")
    sources: List[SourceSpec] = []
    for entry in sources_section:
        if not isinstance(entry, dict):
            raise ValueError("Each source must be a mapping.")
        microrobot = entry.get("microrobot")
        if not microrobot:
            raise ValueError("source.microrobot is required.")
        splits_override = _ensure_list(entry.get("splits")) or dataset_spec.splits
        images_dir = _resolve_optional_path(entry.get("images_dir"))
        labels_dir = _resolve_optional_path(entry.get("labels_dir"))
        sources.append(SourceSpec(microrobot=microrobot, splits=splits_override, images_dir=images_dir, labels_dir=labels_dir))

    augmentations_section = raw.get("augmentations") or []
    augmentations: List[AugmentationPipelineSpec] = []
    for entry in augmentations_section:
        if not isinstance(entry, dict):
            raise ValueError("Each augmentation pipeline must be a mapping.")
        name = entry.get("name")
        if not name:
            raise ValueError("augmentation.name is required.")
        copies = int(entry.get("copies", 1))
        transforms_section = entry.get("transforms") or []
        transforms: List[TransformSpec] = []
        for transform_entry in transforms_section:
            if not isinstance(transform_entry, dict):
                raise ValueError("augmentation.transforms entries must be mappings.")
            t_type = transform_entry.get("type")
            if not t_type:
                raise ValueError("augmentation transform.type is required.")
            probability = float(transform_entry.get("probability", 1.0))
            params = transform_entry.get("params") or {}
            transforms.append(TransformSpec(type=t_type, probability=probability, params=params))
        augmentations.append(AugmentationPipelineSpec(name=name, copies=copies, transforms=transforms))

    return BuilderConfig(dataset=dataset_spec, sources=sources, augmentations=augmentations)


# --------------------------- Augmentation registry ------------------------------


def _ensure_tuple(value: Any, length: int, default: Optional[Sequence[float]] = None) -> Tuple[float, ...]:
    if value is None:
        if default is None:
            raise ValueError("Missing required tuple parameter.")
        return tuple(default)
    if isinstance(value, (list, tuple)):
        if len(value) == length:
            return tuple(float(x) for x in value)
        if len(value) == 1:
            return tuple(float(value[0]) for _ in range(length))
    return tuple(float(value) for _ in range(length))


class AdditiveGaussianNoise(A.ImageOnlyTransform):
    def __init__(
        self,
        sigma_min: float = 0.0,
        sigma_max: float = 0.02,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.sigma_min = float(max(0.0, sigma_min))
        self.sigma_max = float(max(self.sigma_min, sigma_max))

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        noise = np.random.normal(0.0, sigma, size=image.shape).astype(np.float32)
        base = image.astype(np.float32) / 255.0
        augmented = np.clip(base + noise, 0.0, 1.0)
        return (augmented * 255.0).astype(image.dtype)

    def get_transform_init_args_names(self) -> Tuple[str, str]:
        return ("sigma_min", "sigma_max")


class MultiplicativeSpeckleNoise(A.ImageOnlyTransform):
    def __init__(
        self,
        std: float = 0.05,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.std = float(max(0.0, std))

    def apply(self, image: np.ndarray, **params: Any) -> np.ndarray:
        base = image.astype(np.float32) / 255.0
        noise = np.random.normal(0.0, self.std, size=image.shape).astype(np.float32)
        augmented = np.clip(base * (1.0 + noise), 0.0, 1.0)
        return (augmented * 255.0).astype(image.dtype)

    def get_transform_init_args_names(self) -> Tuple[str]:
        return ("std",)


GEOMETRIC_TRANSFORMS = {"affine", "small_affine", "anisotropic_scale"}


def _parse_range(value: Any) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return float(value[0]), float(value[1])
        if len(value) == 1:
            v = float(value[0])
            return -abs(v), abs(v)
    v = float(value)
    return -abs(v), abs(v)


def build_transform(spec: TransformSpec) -> A.BasicTransform:
    t_type = spec.type.lower()
    params = spec.params or {}
    p = float(spec.probability)

    if t_type == "clahe":
        clip_limit = float(params.get("clip_limit", 2.0))
        tile = params.get("tile_grid_size", params.get("tile_grid", 8))
        if isinstance(tile, (list, tuple)):
            tile_grid = tuple(int(x) for x in tile)
        else:
            size = int(tile)
            tile_grid = (size, size)
        return A.CLAHE(clip_limit=clip_limit, tile_grid_size=tile_grid, p=p)

    if t_type in {"speckle_noise", "multiplicative_noise"}:
        std = float(params.get("std", 0.05))
        return MultiplicativeSpeckleNoise(std=std, p=p)

    if t_type in {"gamma", "random_gamma"}:
        gamma_range = params.get("gamma_range")
        if gamma_range:
            gmin, gmax = float(gamma_range[0]), float(gamma_range[1])
        else:
            gmin = float(params.get("gamma_min", 0.9))
            gmax = float(params.get("gamma_max", 1.1))
        gamma_limit = (int(gmin * 100), int(gmax * 100))
        return A.RandomGamma(gamma_limit=gamma_limit, p=p)

    if t_type in {"affine", "small_affine", "anisotropic_scale"}:
        rotate = float(params.get("rotate", 0.0))
        scale = float(params.get("scale", params.get("scale_limit", 0.0)))
        keep_ratio = bool(params.get("keep_ratio", t_type != "anisotropic_scale"))
        translate = params.get("translate_percent", params.get("translate", 0.0))
        shear = params.get("shear")

        affine_kwargs: Dict[str, Any] = {"fit_output": False, "keep_ratio": keep_ratio, "p": p}
        if scale:
            affine_kwargs["scale"] = (max(0.0, 1.0 - scale), 1.0 + scale)
        if translate:
            affine_kwargs["translate_percent"] = _parse_range(translate)
        if rotate:
            affine_kwargs["rotate"] = _parse_range(rotate)
        if shear:
            affine_kwargs["shear"] = _parse_range(shear)
        return A.Affine(**affine_kwargs)

    if t_type in {"motion_blur", "blur"}:
        kernel = int(params.get("kernel", params.get("kernel_size", 3)))
        max_kernel_param = int(params.get("max_kernel", kernel))
        min_kernel = min(kernel, max_kernel_param)
        max_kernel = max(kernel, max_kernel_param)
        # Motion blur expects odd positive kernel sizes
        if min_kernel < 1:
            min_kernel = 1
        if max_kernel < 1:
            max_kernel = 1
        if min_kernel % 2 == 0:
            min_kernel += 1
        if max_kernel % 2 == 0:
            max_kernel += 1
        blur_limit = (min_kernel, max_kernel)
        if blur_limit[0] > blur_limit[1]:
            blur_limit = (blur_limit[1], blur_limit[0])
        return A.MotionBlur(blur_limit=blur_limit, p=p)

    if t_type in {"gaussian_noise", "gauss_noise"}:
        sigma = params.get("sigma")
        if sigma:
            sigma_min, sigma_max = _ensure_tuple(sigma, 2)
        else:
            sigma_min = float(params.get("sigma_min", 0.0))
            sigma_max = float(params.get("sigma_max", 0.02))
        return AdditiveGaussianNoise(sigma_min=sigma_min, sigma_max=sigma_max, p=p)

    raise ValueError(f"Unknown augmentation transform type: {spec.type}")


def build_pipeline(
    spec: AugmentationPipelineSpec,
) -> Optional[Tuple[A.Compose, bool]]:
    if not spec.transforms:
        return None
    transforms = [build_transform(t) for t in spec.transforms]
    modifies_bboxes = any(t.type.lower() in GEOMETRIC_TRANSFORMS for t in spec.transforms)
    compose_kwargs: Dict[str, Any] = {}
    if modifies_bboxes:
        compose_kwargs["bbox_params"] = A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.01,
            clip=True,
            check_each_transform=False,
        )
    pipeline = A.Compose(transforms, **compose_kwargs)
    return pipeline, modifies_bboxes


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
        rng = random.Random(self.shuffle_seed)
        rng.shuffle(samples)
        total = len(samples)
        total_ratio = self.total_ratio or 0.0
        if total_ratio <= 0.0:
            raise ValueError("split ratios must sum to a positive value.")
        allocations: Dict[str, int] = {}
        remaining = total
        splits = self.config.dataset.splits
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

        assigned_counts = {split: 0 for split in splits}
        cursor = 0
        for split in splits:
            count = allocations.get(split, 0)
            for _ in range(count):
                if cursor >= len(samples):
                    break
                samples[cursor].target_split = split
                assigned_counts[split] += 1
                cursor += 1

        while cursor < len(samples):
            last_split = splits[-1]
            samples[cursor].target_split = last_split
            assigned_counts[last_split] += 1
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
    config = load_config(args.config.resolve())
    builder = MultirobotDatasetBuilder(config=config, dry_run=args.dry_run, verbose=not args.quiet)
    builder.run()

if __name__ == "__main__":
    main()
