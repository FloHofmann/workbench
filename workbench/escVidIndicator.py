from pathlib import Path
from workbench.data.preprocess import correct_keyboard_times
import workbench.data.h5data as h5data
import cv2
import numpy as np
import polars as pl
from tkinter import Tk
from tkinter.filedialog import askdirectory
from scipy.io import loadmat
from IPython import embed
import scipy


def mark_vid(out_path: Path, vid_path: Path, keyboard_times):
    radius = 10
    color = (0,0,255)
    thickness = -1
    margin = 20
    dot_window = 0.3

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    position = (margin + radius, margin + radius)

    half_window_frames = int(dot_window * fps / 2)

    dot_frames = set()
    for ts in keyboard_times:
        center_frame = int(round(ts*fps))
        start = max(0, center_frame - half_window_frames)
        end = center_frame + half_window_frames
        dot_frames.update(range(start, end + 1))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in dot_frames:
            cv2.circle(frame, position, radius, color, thickness)
        
        out.write(frame)
        frame_idx +=1
    
    cap.release()
    out.release()
    print("Done, saved to: ", out_path)


path = Path(askdirectory(title='Select Folder'))
#path = Path(r"\\172.25.250.112\burgalossi\lab share\Data\Max\esc_tracking\data\A1")
csv_path = Path(r"\\172.25.250.112\burgalossi\lab share\Data\Max\esc_tracking\data\Recordings_list.csv")
csv = pl.read_csv(csv_path, has_header=True, truncate_ragged_lines=True, separator=';')
mat_list = list(path.glob('*.mat'))
mat_names = [i.stem for i in mat_list]
vid_list = list(path.glob('*.mp4'))
vid_names = [str(i.stem[:6]) for i in vid_list]

mat_map = {p.stem: str(p) for p in mat_list}
vid_map = {}
for p in vid_list:
    stem = p.stem
    prefix = stem[:6]
    vid_map[prefix] = str(p)

cols = csv.columns
csv = csv.with_columns(
    [
        pl.col("video_nr")
            .str.replace("vid","")
            .str.zfill(4)
            .alias("video_id_padded")
    ]
)
csv = csv.with_columns(
    [
        pl.concat_str(
            [pl.lit("FH"), pl.col("video_id_padded")]
        ).alias("video_prefix"),
    ]
)

csv_with_paths = csv.with_columns(
    [
        pl.col("filename")
            .replace(mat_map)
            .alias("mat_path"),
        pl.col("video_prefix")
            .replace(vid_map)
            .alias("vid_path")
    ]
)

filt_csv = csv_with_paths.filter(
    pl.col(cols[0]).is_in(mat_names)
)

for mat_p, vid_p in zip(filt_csv['mat_path'], filt_csv['vid_path']):
    print(Path(mat_p).stem)
    mat = h5data.loadmat(mat_p)
    keyboard_channel = mat.raw_data['Ch31']
    times = keyboard_channel['times']
    times = np.array(times[0][1:-1]*25000) # spike2 sampling rate
    vid_folder = Path(vid_p).parent
    prefix = Path(vid_p).stem[:6]
    filename = prefix+"_marked.mp4"
    out_path = Path(vid_folder, filename)
    mark_vid(out_path, Path(vid_p), times)
