import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image_label_pairs(images_dir: Path, labels_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    """
    Collect matching image/label pairs.
    Images without a label are reported but skipped.
    """
    pairs: List[Tuple[Path, Path]] = []
    missing_labels: List[Path] = []

    for image_path in images_dir.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS or not image_path.is_file():
            continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path)

    return pairs, missing_labels


def split_indices(total: int, train_ratio: float, val_ratio: float) -> Dict[str, slice]:
    """
    Compute index slices for train/val/test splits.
    """
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": slice(0, train_end),
        "val": slice(train_end, val_end),
        "test": slice(val_end, total),
    }


def prepare_output_dirs(output_dir: Path) -> None:
    """
    Create YOLO-style directory layout:

        output_dir/
          train/
            images/
            labels/
          val/
            images/
            labels/
          test/
            images/
            labels/
    """
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_split(pairs: List[Tuple[Path, Path]], output_dir: Path, slices: Dict[str, slice]) -> Dict[str, int]:
    """
    Copy files into split folders based on provided slices.
    """
    copied: Dict[str, int] = {}
    for split, slc in slices.items():
        items = pairs[slc]
        for image_path, label_path in items:
            shutil.copy2(image_path, output_dir / split / "images" / image_path.name)
            shutil.copy2(label_path, output_dir / split / "labels" / label_path.name)
        copied[split] = len(items)
    return copied


def split_yolo_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, object]:
    """ 
    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing YOLO .txt labels (same stem as images).
        output_dir: Destination directory for the split dataset.
        train_ratio: Fraction of samples for the train split.
        val_ratio: Fraction of samples for the val split.
        seed: Random seed for reproducible shuffling.

    Returns:
        Dict with statistics:
        {
            "num_pairs": int,
            "num_missing_labels": int,
            "copied_counts": {"train": int, "val": int, "test": int},
            "output_dir": Path,
        }
    """
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Train and val ratios must be > 0 and sum to less than 1. Remainder is used for test.")

    images_dir = images_dir.resolve()
    labels_dir = labels_dir.resolve()
    output_dir = output_dir.resolve()

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Images dir ({images_dir}) or labels dir ({labels_dir}) does not exist.")

    pairs, missing_labels = find_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        raise ValueError("No image/label pairs were found. Check the provided directories and extensions.")

    random.Random(seed).shuffle(pairs)
    prepare_output_dirs(output_dir)
    slices = split_indices(len(pairs), train_ratio, val_ratio)
    copied_counts = copy_split(pairs, output_dir, slices)

    return {
        "num_pairs": len(pairs),
        "num_missing_labels": len(missing_labels),
        "copied_counts": copied_counts,
        "output_dir": output_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a YOLO-style dataset (images + labels) into train/val/test folders.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("images"),
        help="Directory containing images to split.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("labels"),
        help="Directory containing YOLO .txt labels that match the image stems.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("splits"),
        help="Destination directory where train/val/test folders will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of samples for the train split.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of samples for the val split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to keep the split reproducible.")
    args = parser.parse_args()

    stats = split_yolo_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Images with labels: {stats['num_pairs']}")
    print(f"Missing labels skipped: {stats['num_missing_labels']}")
    for split, count in stats["copied_counts"].items():
        print(f"{split}: {count}")
    print(f"Output written to: {stats['output_dir']}")


if __name__ == "__main__":
    main()
