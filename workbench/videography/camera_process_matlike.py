from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import cv2 as cv
import numpy as np


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate NaNs in a 1D array in-place and return it.
    Leaves leading/trailing NaNs filled with nearest valid value (pad).
    """
    x = np.arange(arr.size)
    mask = np.isfinite(arr)
    if mask.sum() == 0:
        return arr  # all NaN; leave as-is
    # interpolate interior
    arr[~mask] = np.interp(x[~mask], x[mask], arr[mask])
    # pad edges if needed
    first, last = np.flatnonzero(mask)[[0, -1]]
    if first > 0:
        arr[:first] = arr[first]
    if last < arr.size - 1:
        arr[last + 1:] = arr[last]
    return arr


def _kasa_circle_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Algebraic circle fit (Kåsa method). Robust enough for well-formed circular data.
    Returns (xc, yc, R). R is the mean radius from the fitted center.
    """
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    # Solve A*[xc, yc, c]^T = b  (least squares)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc, c = sol
    R = float(np.mean(np.sqrt((x - xc) ** 2 + (y - yc) ** 2)))
    return float(xc), float(yc), R


def _project_to_circle(
    x: np.ndarray, y: np.ndarray, xc: float, yc: float, R: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project each (x, y) onto the circle centered at (xc, yc) with radius R.
    """
    vx = x - xc
    vy = y - yc
    mag = np.hypot(vx, vy)
    # Avoid divide-by-zero; if mag==0, leave as center+R along +x
    mag_safe = np.where(mag == 0, 1.0, mag)
    x_proj = xc + vx / mag_safe * R
    y_proj = yc + vy / mag_safe * R
    # For points exactly at center, put them at angle 0 on circle
    at_center = mag == 0
    if np.any(at_center):
        x_proj[at_center] = xc + R
        y_proj[at_center] = yc
    return x_proj, y_proj


def _wrap_to_360(deg: np.ndarray) -> np.ndarray:
    """
    MATLAB-like wrapTo360 for degrees.
    """
    wrapped = np.mod(deg, 360.0)
    wrapped[wrapped < 0] += 360.0
    return wrapped


def track_platform(
    path: Path,
    *,
    threshold_value: int = 10,
    blur_kernel: Tuple[int, int] = (5, 5),
    phase_offset_deg: float = -45.0,       # mirrors MATLAB's -45°
    reverse_direction: bool = False,        # mirrors commented MATLAB option
    min_detection_rate: float = 0.1,
    binary_threshold_max: int = 255,
    # optional boolean mask (H,W): True=keep
    roi_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Process an .avi file and return:
      angles (deg, [0,360)),
      nframes,
      x, y (projected to best-fit circle, twice, like MATLAB).

    Differences vs. your earlier Python:
      - Fits circle and projects to it (twice), like MATLAB.
      - Uses wrapTo360 + phase offset (default -45°) instead of +180 shift.
      - Computes detection rate and raises if below threshold.
      - Interpolates missing frames before fitting, like MATLAB's interpolateFrames.

    Parameters
    ----------
    path : Path
        Path to the .avi file.
    threshold_value : int
        Binary threshold for LED detection (on blurred grayscale).
    blur_kernel : (int,int)
        Gaussian blur kernel size.
    phase_offset_deg : float
        Fixed phase offset applied after wrapTo360 (default -45°).
    reverse_direction : bool
        If True, flip angle direction (equivalent to Angles = -Angles in MATLAB).
    min_detection_rate : float
        Minimum fraction of frames that must have a detected centroid.
    binary_threshold_max : int
        Max value used in cv.threshold.
    roi_mask : np.ndarray[H,W], optional
        If provided, a boolean mask applied before contour search (non-True pixels cleared).
    """
    cam = cv.VideoCapture(str(path))
    nframes: int = int(cam.get(cv.CAP_PROP_FRAME_COUNT))
    led_coords = np.full((2, nframes), np.nan, dtype=np.float32)

    frame_idx = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if roi_mask is not None:
            # ensure mask shape compatibility (resize if needed)
            if roi_mask.shape != gray.shape:
                roi_resized = cv.resize(
                    roi_mask.astype(np.uint8) *
                    255, (gray.shape[1], gray.shape[0]),
                    interpolation=cv.INTER_NEAREST,
                ) > 0
            else:
                roi_resized = roi_mask
        else:
            roi_resized = None

        gray_blur = cv.GaussianBlur(gray, blur_kernel, 0)
        _, thresh = cv.threshold(
            gray_blur, threshold_value, binary_threshold_max, cv.THRESH_BINARY)

        if roi_resized is not None:
            thresh = np.where(roi_resized, thresh, 0).astype(np.uint8)

        contours, _ = cv.findContours(
            thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv.contourArea)
            M = cv.moments(largest)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                led_coords[:, frame_idx] = (cx, cy)

        frame_idx += 1

    cam.release()

    # If the video reported more frames than read (rare), trim arrays
    if frame_idx < nframes:
        led_coords = led_coords[:, :frame_idx]
        nframes = frame_idx

    # Detection rate check (like MATLAB's guard rail)
    detected = ~np.isnan(led_coords).any(axis=0)
    detection_rate = float(detected.mean()) if nframes > 0 else 0.0
    if detection_rate < min_detection_rate:
        raise ValueError(
            f"Detection rate {detection_rate:.3f} < {min_detection_rate:.3f}. "
            "Check LED visibility and parameters."
        )

    # Interpolate missing frames independently on x and y
    x_values = led_coords[0, :].astype(np.float64)
    y_values = led_coords[1, :].astype(np.float64)
    x_values = _interpolate_nans(x_values)
    y_values = _interpolate_nans(y_values)

    # === First fit & projection to circle ===
    xc, yc, R = _kasa_circle_fit(x_values, y_values)
    x_proj, y_proj = _project_to_circle(x_values, y_values, xc, yc, R)

    # === Second fit & projection (like MATLAB doing it again) ===
    xc2, yc2, R2 = _kasa_circle_fit(x_proj, y_proj)
    x_final, y_final = _project_to_circle(x_proj, y_proj, xc2, yc2, R2)

    # === Angles with wrapTo360 and phase offset (MATLAB style) ===
    angles = np.degrees(np.arctan2(y_final - yc2, x_final - xc2))
    if reverse_direction:
        angles = -angles  # mirrors the commented MATLAB option
    angles = _wrap_to_360(angles + phase_offset_deg)

    return angles.astype(np.float32), nframes, x_final.astype(np.float32), y_final.astype(np.float32)


# If you want a quick smoke test, you could add:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    angles, frames, _, _ = track_platform(Path(
        r"\\172.25.250.112\burgalossi\lab share\Data\Florian\ADN\direction check_clockwise_frontleft_to_frontright.avi"))
    plt.plot(angles, c='r')
    plt.show()
