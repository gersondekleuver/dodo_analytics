from pathlib import Path

from ultralytics import YOLO
from datetime import date
import argparse

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
TODAY = str(date.today())


def iter_videos(directory: Path):
    for extension in VIDEO_EXTENSIONS:
        yield from directory.glob(f"*{extension}")


def main():
    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument(
        "--model_path", "-m", type=str, required=True, help="Model path."
    )
    parser.add_argument(
        "--input_folder", "-i", type=str, required=True, help="input folder."
    )
    parser.add_argument(
        "--output_folder", "-o", type=str, required=True, help="output folder"
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    if not input_folder.exists():
        raise FileNotFoundError(f"DJI folder not found: {input_folder}")

    videos = sorted(video for video in iter_videos(input_folder) if video.is_file())

    if not videos:
        raise FileNotFoundError(f"No videos found in {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    for video_path in videos:
        run_name = video_path.stem + TODAY
        print(f"Processing {video_path} -> {output_folder / run_name}")
        model.predict(
            source=str(video_path),
            save=True,
            project=str(output_folder),
            name=run_name,
            exist_ok=True,
        )


if __name__ == "__main__":
    main()
