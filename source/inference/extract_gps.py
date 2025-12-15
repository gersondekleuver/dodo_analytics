import subprocess
import csv
import io
from pathlib import Path
import pandas as pd


def extract_gps_from_folder(folder: str) -> pd.DataFrame:
    """
    Extract GPS data from all video files in a folder using exiftool.
    Returns a pandas DataFrame with columns:
    SourceFile, SampleTime, GPSLatitude, GPSLongitude, GPSAltitude, GPSSpeed
    """
    folder_path = Path(folder)

    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder} is not a valid directory")

    # exiftool options:
    # -r   : recurse into subfolders (remove if you don't want that)
    # -ee  : extract embedded metadata (per-frame/per-sample)
    # -n   : numeric values
    # -csv : CSV output
    cmd = [
        "exiftool",
        "-api", "largefilesupport=1",
        "-r",
        "-ee",
        "-n",
        "-csv",
        "-SampleTime",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSAltitude",
        "-GPSSpeed",
        str(folder_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"exiftool failed ({result.returncode}):\n{result.stderr}"
        )

    csv_text = result.stdout
    if not csv_text.strip():
        raise ValueError("No CSV output from exiftool. "
                         "Do the files contain GPS/telemetry data?")

    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)

    if not rows:
        raise ValueError("No GPS rows found in the folder.")

    df = pd.DataFrame(rows)

    # Convert numeric columns where present
    for col in ["GPSLatitude", "GPSLongitude", "GPSAltitude", "GPSSpeed", "SampleTime"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract GPS data from all DJI videos in a folder using exiftool"
    )
    parser.add_argument("folder", help="Folder containing video files (MP4/MOV etc.)")
    parser.add_argument(
        "--out",
        help="Optional CSV output path (default: gps_all_videos.csv in the folder)",
        default=None,
    )
    args = parser.parse_args()

    df = extract_gps_from_folder(args.folder)
    print(df.head())

    out_path = args.out or str(Path(args.folder) / "gps_all_videos.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved GPS data to {out_path}")
