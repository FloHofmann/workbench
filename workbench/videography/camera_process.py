"""this module is used to process the camera inputs i.e. pupil tracking or
platform tracking"""

from pathlib import Path

import cv2 as cv
import numpy as np


def track_platform(path: Path) -> tuple[np.array, int, np.array, np.array]:
    """Taking a pathlib reference to the to-be-processed .avi file, returning
    a vector holding the angular
    information throughout the recording.
    Input is a string or a pathlib Path to the .avi file to track.
    Output:
        0. angles -> the angle of the LED for each frame with respect to the
        circles center
        1. nframes -> total number of frames
        2. x -> the x values of each angle on a cartesian coordinate system
        3. y -> the y values of each angle on a cartesian coordinate system
    """
    # Constants
    THRESHOLD_VALUE = 10
    BLUR_KERNEL_SIZE = (5, 5)

    # Read the video from path
    cam = cv.VideoCapture(str(path))
    nframes: int = int(cam.get(cv.CAP_PROP_FRAME_COUNT))
    led_coords = np.full((2, nframes), np.nan, dtype=np.float32)
    frame_idx = 0

    while 1:
        ret, frame = cam.read()
        if not ret:
            break

        grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        grey_blur = cv.GaussianBlur(grey, BLUR_KERNEL_SIZE, 0)
        _, thresh_grey = cv.threshold(grey_blur, THRESHOLD_VALUE, 255,
                                      cv.THRESH_BINARY)
        # Find contours
        contours, _ = cv.findContours(
            thresh_grey, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour (assumed to be the LED)
            largest_contour = max(contours, key=cv.contourArea)
            M = cv.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                led_coords[:, frame_idx] = [cx, cy]

        frame_idx += 1

    cam.release()
# Threshold filtering (same as original)
    led_coords[:, led_coords[0, :] < 210] = np.nan
    led_coords[:, led_coords[1, :] > 400] = np.nan

    # Interpolation of missing (nan) values
    for i in range(2):
        nan_indices = np.isnan(led_coords[i, :])
        non_nan_indices = ~nan_indices
        if np.sum(non_nan_indices) == 0:
            # If all are NaN, skip interpolation to avoid errors
            continue
        led_coords[i, nan_indices] = np.interp(
            np.where(nan_indices)[0],
            np.where(non_nan_indices)[0],
            led_coords[i, non_nan_indices]
        )

    # Calculate center of the circle
    x_values = led_coords[0, :]
    y_values = led_coords[1, :]
    xmid = (np.max(x_values) + np.min(x_values)) / 2
    ymid = (np.max(y_values) + np.min(y_values)) / 2

    # Calculate angles relative to center
    dx = x_values - xmid
    dy = y_values - ymid
    angles = np.round(np.degrees(np.arctan2(dy, dx)) + 180, 3)
    return angles, nframes, x_values, y_values


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from pathlib import Path
    angles, frames, _, _ = track_platform(Path(
        r"\\172.25.250.112\burgalossi\lab share\Data\Florian\Behavior\RH13\data\FH0001 24-10-17 10-20-35.avi"))
    plt.plot(angles, c='r')
    plt.show()
