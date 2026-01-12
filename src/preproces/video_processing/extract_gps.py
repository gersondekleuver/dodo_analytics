import subprocess
import re
from pathlib import Path
import pandas as pd
import cv2
from scipy.interpolate import interp1d


# =========================================================
# TIME UTIL
# =========================================================
def srt_time_to_seconds(t: str) -> float:
    h, m, s_ms = t.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# =========================================================
# EXTRACT EMBEDDED DJI SRT
# =========================================================
def extract_embedded_srt(video: Path, srt_out: Path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-map", "0:2",
        "-c:s", "srt",
        str(srt_out),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# =========================================================
# PARSE DJI MINI 3 SRT
# GPS (lon, lat, alt), H.S x m/s, V.S y m/s
# =========================================================
def parse_dji_srt(srt_path: Path) -> pd.DataFrame:
    rows = []
    lines = srt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    gps_pattern = re.compile(
        r"GPS\s*\(\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*,\s*([-0-9.]+)\s*\)",
        re.IGNORECASE,
    )
    hs_pattern = re.compile(r"H\.S\s*([-0-9.]+)", re.IGNORECASE)
    vs_pattern = re.compile(r"V\.S\s*([-0-9.]+)", re.IGNORECASE)

    i = 0
    while i < len(lines):
        if "-->" in lines[i]:
            start_time = lines[i].split(" --> ")[0].strip()
            t = srt_time_to_seconds(start_time)

            if i + 1 < len(lines):
                text = lines[i + 1]

                gps_match = gps_pattern.search(text)
                if gps_match:
                    lon, lat, alt = map(float, gps_match.groups())

                    hs = None
                    vs = None

                    m_hs = hs_pattern.search(text)
                    if m_hs:
                        hs = float(m_hs.group(1))

                    m_vs = vs_pattern.search(text)
                    if m_vs:
                        vs = float(m_vs.group(1))

                    rows.append({
                        "SampleTime": t,
                        "GPSLatitude": lat,
                        "GPSLongitude": lon,
                        "GPSAltitude": alt,
                        "HorizontalSpeed": hs,
                        "VerticalSpeed": vs,
                    })
        i += 1

    if not rows:
        raise RuntimeError(f"No GPS data parsed from {srt_path}")

    return pd.DataFrame(rows)


# =========================================================
# FRAME TIMES
# =========================================================
def extract_frame_times(video_path: Path) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    idx = 0

    while cap.isOpened():
        ret, _ = cap.read()
        if not ret:
            break
        frames.append({
            "FrameIndex": idx,
            "FrameTime": idx / fps,
        })
        idx += 1

    cap.release()
    return pd.DataFrame(frames)


# =========================================================
# INTERPOLATE GPS TO EVERY FRAME
# =========================================================
def interpolate_gps(gps: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    interp_lat = interp1d(
        gps["SampleTime"], gps["GPSLatitude"], fill_value="extrapolate"
    )
    interp_lon = interp1d(
        gps["SampleTime"], gps["GPSLongitude"], fill_value="extrapolate"
    )
    interp_alt = interp1d(
        gps["SampleTime"], gps["GPSAltitude"], fill_value="extrapolate"
    )

    frames["GPSLatitude"] = interp_lat(frames["FrameTime"])
    frames["GPSLongitude"] = interp_lon(frames["FrameTime"])
    frames["GPSAltitude"] = interp_alt(frames["FrameTime"])

    if "HorizontalSpeed" in gps.columns:
        interp_hs = interp1d(
            gps["SampleTime"], gps["HorizontalSpeed"], fill_value="extrapolate"
        )
        frames["HorizontalSpeed"] = interp_hs(frames["FrameTime"])

    if "VerticalSpeed" in gps.columns:
        interp_vs = interp1d(
            gps["SampleTime"], gps["VerticalSpeed"], fill_value="extrapolate"
        )
        frames["VerticalSpeed"] = interp_vs(frames["FrameTime"])

    return frames


# =========================================================
# PROCESS SINGLE VIDEO
# =========================================================
def process_single_video(video: Path, output_dir: Path):
    print(f"▶ Processing {video.name}")

    srt = video.with_suffix(".srt")
    if not srt.exists():
        print("  ↳ Extracting embedded DJI subtitles")
        extract_embedded_srt(video, srt)

    gps = parse_dji_srt(srt)
    frames = extract_frame_times(video)
    frames = interpolate_gps(gps, frames)

    frames["SourceFile"] = video.name

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{video.stem}.csv"
    frames.to_csv(out_csv, index=False)

    print(f"Wrote {out_csv}")


# =========================================================
# PROCESS FOLDER
# =========================================================
def process_folder(input_folder: Path, output_folder: Path):
    videos = list(input_folder.rglob("*.mp4"))
    if not videos:
        raise RuntimeError("No MP4 files found")

    for video in videos:
        process_single_video(video, output_folder)


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract per-frame GPS from DJI Mini 3 videos (one CSV per video)"
    )
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-o", "--output_folder", required=True)

    args = parser.parse_args()
    process_folder(Path(args.input_folder), Path(args.output_folder))
