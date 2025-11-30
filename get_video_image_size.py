#!/usr/bin/env python3
from pathlib import Path
import cv2

video_dir = Path("DJI_flights")

for video_path in sorted(video_dir.glob("*.mp4")):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Failed to open {video_path.name}")
        continue
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"{video_path.name}: {width}x{height}")
    cap.release()