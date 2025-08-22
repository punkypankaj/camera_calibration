#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2
import yaml
import numpy as np

yaml_path = sys.argv[1] if len(sys.argv) > 1 else "calibrationfiles/calibration_matrix.yaml"

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)

mtx  = np.array(data["camera_matrix"], dtype="float32")
dist = np.array(data["dist_coeff"], dtype="float32")
w, h = int(data["image_width"]), int(data["image_height"])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

while True:
    ok, frame = cap.read()
    if not ok:
        break
    und = cv2.undistort(frame, mtx, dist, None, new_mtx)
    cv2.imshow("undistorted (press q to quit)", und)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
