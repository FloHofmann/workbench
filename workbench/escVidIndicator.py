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


def mark_vid(
    out_path: Path,
    vid_path: Path,
    stim_samples: np.ndarray,
    sample_rate: float | None = None,
    *,
    start_offset: float = 0.3,
    vid_duration: float | None = None,
    duration: float = 0.3,
    border_color: tuple = (0, 0, 255),
    thickness: int = 12,
    samples_are_frames: bool = False,
):
    """Create a copy of vid_path saved at out_path where the video border
    turns red for `duration` seconds starting at stim_time + start_offset.

    stim_samples can be:
      - an array of times in seconds (default),
      - an array of sample indices with sample_rate provided,
      - or an array of frame indices with samples_are_frames=True.
    """

    if vid_path.resolve() == out_path.resolve():
        raise ValueError(
            f'Input and output videos are the same file: {vid_path}')
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open input video: %s" % vid_path)

    fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # convert samples -> seconds
    sarr = np.asarray(stim_samples, dtype=float).flatten()
    if samples_are_frames:
        stim_seconds = sarr / max(1.0, fps)
    elif sample_rate is not None:
        stim_seconds = sarr / float(sample_rate)
    else:
        stim_seconds = sarr

    if vid_duration:
        fps = total_frames/vid_duration

    # compute frame ranges
    ranges = []
    for t in stim_seconds:
        start_s = float(t)
        end_s = start_s + float(duration)
        if end_s <= start_s:
            continue
        start_frame = int(round(start_s * fps))
        end_frame = int(round(end_s * fps))
        if total_frames and start_frame > total_frames:
            # start beyond video length -> skip
            print(f"Video {
                  vid_path} shorter than the corresponding mat file. Cannot be processed.")
            continue
        ranges.append((max(0, start_frame), max(0, end_frame)))

    # if we know total_frames, build boolean mask for quick lookup
    mask = None
    if total_frames > 0:
        mask = np.zeros(total_frames, dtype=bool)
        for a, b in ranges:
            b = min(b, total_frames - 1)
            if b >= a:
                mask[a:b + 1] = True

    # sanity check
    print("FPS:", fps, "Total frames:", total_frames)
    print("First few stim seconds:", stim_seconds[:5])
    print("First few ranges:", ranges[:5])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(
            f"Could not open ooutput video for writing: {out_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        on = False
        if mask is not None:
            if 0 <= frame_idx < len(mask) and mask[frame_idx]:
                on = True
        else:
            # fallback: check ranges directly
            for a, b in ranges:
                if a <= frame_idx <= b:
                    on = True
                    break

        if on:
            # draw border (BGR) rectangle
            cv2.rectangle(
                frame, (0, 0), (width - 1, height - 1), border_color, thickness
            )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("Done, saved to:", out_path)


def mark_video_selection():
    path = Path(askdirectory(title='Select Folder'))
    # path = Path(r"\\172.25.250.112\burgalossi\lab share\Data\Max\esc_tracking\data\A1")
    csv_path = Path(
        r"\\172.25.250.112\burgalossi\lab share\Data\Max\esc_tracking\data\Recordings_list.csv")
    csv = pl.read_csv(csv_path, has_header=True,
                      truncate_ragged_lines=True, separator=';')
    mat_list = list(path.glob('*.mat'))
    mat_names = [i.stem.strip() for i in mat_list]

    mat_map = {p.stem: str(p) for p in mat_list}
    vid_list = [p for p in path.glob(
        '*.mp4') if not p.stem.endswith('_marked')]
    vid_names = [str(i.stem[:6]).strip() for i in vid_list]

    vid_map = {}
    for p in vid_list:
        stem = p.stem
        prefix = stem[:6]
        vid_map[prefix] = str(p)

    cols = csv.columns
    csv = csv.with_columns(
        [
            pl.col("video_nr")
            .str.replace("vid", "")
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
        pl.col(cols[1]).is_in(mat_names)
    )

    for mat_p, vid_p in zip(filt_csv['mat_path'], filt_csv['vid_path']):
        from scipy.signal import find_peaks
        print(Path(mat_p).stem)
        mat = h5data.loadmat(mat_p)
        keyboard_channel = mat.raw_data['Ch31']
        sound_channel = mat.raw_data['Ch5']
        click_channel = mat.raw_data['Ch4']
        ttl_channel = mat.raw_data['Ch3']['values']
        diffs = np.diff(ttl_channel[:])
        pks = find_peaks(diffs, height=4)
        print(f"{len(pks[0])} TTLs were given")

        # split the keyboards pressed and their times for keyboard correction
        handlers = {
            'w': {'config': click_channel},
            'a': {'config': sound_channel},
            'e': {'config': click_channel}
        }
        times = keyboard_channel['times'][0]
        vid_duration = times[-1] - times[0]
        pre_time = times[0]
        times = times[1:-1]  # keyboard times/stimulus times
        codes = list(''.join(map(chr, keyboard_channel['codes'][0])))
        codes = codes[1:-1]

        result = np.array([
            correct_keyboard_times(handlers[l]['config'], [v])
            for v, l in zip(times, codes)
        ])

        # korregieren der stimulus time
        times = result - pre_time

        vid_folder = Path(vid_p).parent
        prefix = Path(vid_p).stem[:6]
        filename = prefix+"_marked.mp4"
        out_path = Path(vid_folder, filename)
        mark_vid(out_path, Path(vid_p), times, sample_rate=None,
                 samples_are_frames=False, vid_duration=vid_duration)


if __name__ == "__main__":
    mark_video_selection()
