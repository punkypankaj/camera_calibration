#!/usr/bin/env python3
"""
Camera calibration for Raspberry Pi Camera V2 (Python 3).
- Preserves original CLI: --mm --width --height
- Extras (optional): --rows --cols --image_dir OR --live, --device, --samples, --output, --save_preview
- Outputs calibrationfiles/calibration_matrix.yaml (same path as common repos)
"""

import argparse
import os
import sys
import glob
import time
import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Camera calibration (Python 3, OpenCV 4).")
    p.add_argument("--mm", type=float, required=True,
                   help="Checkerboard square size in millimeters (e.g., 26).")
    p.add_argument("--width", type=int, required=True,
                   help="Image width used for calibration (e.g., 640).")
    p.add_argument("--height", type=int, required=True,
                   help="Image height used for calibration (e.g., 480).")

    # Optional (defaults are common inner-corner counts for a 9x6 board)
    p.add_argument("--rows", type=int, default=6,
                   help="Checkerboard inner corners per column (rows). Default 6.")
    p.add_argument("--cols", type=int, default=9,
                   help="Checkerboard inner corners per row (cols). Default 9.")

    # Choose images-from-folder OR live capture
    p.add_argument("--image_dir", type=str, default="",
                   help="Folder of calibration images (jpg/png).")
    p.add_argument("--live", action="store_true",
                   help="Capture calibration images live from a camera.")

    # Live capture options
    p.add_argument("--device", type=int, default=0,
                   help="cv2.VideoCapture device index. Default 0.")
    p.add_argument("--samples", type=int, default=20,
                   help="Target number of good boards to collect if --live is used.")

    # Output
    p.add_argument("--output", type=str,
                   default="calibrationfiles/calibration_matrix.yaml",
                   help="Output YAML path.")
    p.add_argument("--save_preview", action="store_true",
                   help="Also save an undistortion preview image.")

    return p.parse_args()


def ensure_parent_dir(path_str: str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def collect_images_from_dir(image_dir: str):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    base = Path(image_dir)
    for e in exts:
        paths += glob.glob(str(base / e))
    paths.sort()
    return paths


def collect_images_live(device: int, width: int, height: int,
                        want: int, rows: int, cols: int):
    print("[i] Live capture mode")
    print("    - Press 's' to save when corners are detected (green).")
    print("    - Press 'q' to finish.")

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[!] Could not open camera index {device}")
        sys.exit(1)

    # Try to set desired resolution (works for UVC; libcamera may ignore)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    out_dir = Path("captured_images")
    out_dir.mkdir(exist_ok=True)

    saved = 0
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FAST_CHECK)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[!] Camera read failed.")
            break

        view = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if found:
            # refine
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
            cv2.drawChessboardCorners(view, (cols, rows), corners, found)
            msg = "Corners OK - press 's' to save"
            color = (0, 255, 0)
        else:
            msg = "Show checkerboard clearly..."
            color = (0, 0, 255)

        cv2.putText(view, msg, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        cv2.putText(view, f"{saved}/{want} saved", (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("calibration capture", view)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and found:
            fname = out_dir / f"img_{saved:03d}.png"
            cv2.imwrite(str(fname), frame)
            saved += 1
            print(f"[+] saved {fname}")
            if saved >= want:
                break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return collect_images_from_dir(out_dir)


def calibrate_from_images(paths, width, height, rows, cols, square_mm):
    if len(paths) < 3:
        print(f"[!] Need at least 3 images, got {len(paths)}")
        sys.exit(1)

    # Prepare one board's 3D points (Z=0 plane)
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)  # scale only affects extrinsics

    objpoints = []  # 3D points per view
    imgpoints = []  # 2D points per view

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    used = 0
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[ ] could not read {p}")
            continue

        img = cv2.resize(img, (width, height))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if not found:
            print(f"[ ] corners not found in {p}")
            continue

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)
        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1

    if used < 3:
        print(f"[!] Too few valid boards: {used}")
        sys.exit(1)

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (width, height), None, None
    )

    # Mean reprojection error (RMS)
    tot_err = 0.0
    tot_pts = 0
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
        tot_err += err * err
        tot_pts += len(proj)
    rms = float(np.sqrt(tot_err / tot_pts))

    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))

    return {
        "rms": rms,
        "camera_matrix": mtx,
        "dist_coeff": dist,
        "new_camera_matrix": new_mtx,
        "roi": roi,
        "used_images": int(used),
    }


def save_yaml(out_path, width, height, rows, cols, square_mm, calib):
    ensure_parent_dir(out_path)
    data = {
        "image_width": int(width),
        "image_height": int(height),
        "pattern_rows": int(rows),
        "pattern_cols": int(cols),
        "square_size_mm": float(square_mm),
        "rms_reprojection_error": float(calib["rms"]),
        "camera_matrix": np.asarray(calib["camera_matrix"]).tolist(),
        "dist_coeff": np.asarray(calib["dist_coeff"]).ravel().tolist(),
        "new_camera_matrix": np.asarray(calib["new_camera_matrix"]).tolist(),
        "roi": [int(x) for x in calib["roi"]],
        "file_date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f)
    print(f"[✓] wrote {out_path}")


def save_preview(paths, width, height, calib):
    if not paths:
        return
    img = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if img is None:
        return
    img = cv2.resize(img, (width, height))
    und = cv2.undistort(
        img,
        calib["camera_matrix"],
        calib["dist_coeff"],
        None,
        calib["new_camera_matrix"],
    )
    out = Path("calibrationfiles/preview_undistorted.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), und)
    print(f"[i] preview saved to {out}")


def main():
    args = parse_args()

    if args.image_dir:
        img_paths = collect_images_from_dir(args.image_dir)
        print(f"[i] Found {len(img_paths)} images in {args.image_dir}")
    elif args.live:
        img_paths = collect_images_live(args.device, args.width, args.height,
                                        args.samples, args.rows, args.cols)
        print(f"[i] Collected {len(img_paths)} images.")
    else:
        print("[!] Provide --image_dir or use --live to capture.")
        sys.exit(1)

    calib = calibrate_from_images(
        img_paths, args.width, args.height, args.rows, args.cols, args.mm
    )
    print(f"[i] RMS reprojection error: {calib['rms']:.4f} (lower is better)")
    save_yaml(args.output, args.width, args.height, args.rows, args.cols, args.mm, calib)
    if args.save_preview:
        save_preview(img_paths, args.width, args.height, calib)


if __name__ == "__main__":
    main()
