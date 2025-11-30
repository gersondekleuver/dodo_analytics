import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_image_label_pairs(images_dir: Path, labels_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    """
    Collect matching image/label pairs. Images without a label are reported but skipped.
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
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {"train": slice(0, train_end), "val": slice(train_end, val_end), "test": slice(val_end, total)}


def prepare_output_dirs(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def copy_split(pairs: List[Tuple[Path, Path]], output_dir: Path, slices: Dict[str, slice]) -> Dict[str, int]:
    copied = {}
    for split, slc in slices.items():
        items = pairs[slc]
        for image_path, label_path in items:
            shutil.copy2(image_path, output_dir / split / "images" / image_path.name)
            shutil.copy2(label_path, output_dir / split / "labels" / label_path.name)
        copied[split] = len(items)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split DroneWaste VQ images/labels into train/val/test folders with YOLO-style layout.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("datasets/bbox/preprocessed_yolo_formatted_files/dronewaste_V1/images"),
        help="Directory containing images to split.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("datasets/bbox/preprocessed_yolo_formatted_files/dronewaste_V1/labels"),
        help="Directory containing YOLO txt labels that match the image stems.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/bbox/preprocessed_yolo_formatted_files/dronewaste_V1_splits"),
        help="Destination directory where train/val/test folders will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fraction of samples for the train split.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction of samples for the val split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to keep the split reproducible.")
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Train and val ratios must be > 0 and sum to less than 1. Remainder is used for test.")

    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Images dir ({images_dir}) or labels dir ({labels_dir}) does not exist.")

    pairs, missing_labels = find_image_label_pairs(images_dir, labels_dir)
    if not pairs:
        raise ValueError("No image/label pairs were found. Check the provided directories and extensions.")

    random.Random(args.seed).shuffle(pairs)
    prepare_output_dirs(output_dir)
    slices = split_indices(len(pairs), args.train_ratio, args.val_ratio)
    copied_counts = copy_split(pairs, output_dir, slices)

    print(f"Images with labels: {len(pairs)}")
    print(f"Missing labels skipped: {len(missing_labels)}")
    for split, count in copied_counts.items():
        print(f"{split}: {count}")
    print(f"Output written to: {output_dir}")


if __name__ == "__main__":
    main()
