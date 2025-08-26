#!/usr/bin/env python3
import os
import json
import csv
import subprocess
import tempfile
import sys
from pathlib import Path
import numpy as np
import cv2

# ---------- Utilities ----------


def read_gray(frame):
    if frame is None:
        raise RuntimeError("Frame is None.")
    if len(frame.shape) == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def build_background(path, sample_every=300, max_samples=50):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    step = max(1, min(sample_every, max(1, total // max_samples)))
    idx = 0
    while idx < total and len(frames) < max_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            break
        frames.append(read_gray(f))
        idx += step
    cap.release()
    if not frames:
        raise RuntimeError("Could not sample frames for background.")
    return np.median(np.stack(frames, 0), 0).astype(np.uint8)


def auto_detect_shelter(gray_bg, min_rect_area_frac=0.02):
    inv = cv2.bitwise_not(gray_bg)
    th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), 2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray_bg.shape
    min_area = min_rect_area_frac * (H*W)
    best, best_area = None, 0
    for c in cnts:
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        area = cv2.contourArea(approx)
        if area > best_area and area > min_area and len(approx) in (4, 5):
            best, best_area = approx, area
    return cv2.boundingRect(best) if best is not None else None


def manual_roi_from_first_frame(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame for manual ROI.")
    r = cv2.selectROI("Select shelter ROI (ENTER to confirm)", f, False, True)
    cv2.destroyWindow("Select shelter ROI (ENTER to confirm)")
    x, y, w, h = map(int, r)
    return (x, y, w, h)


def has_cuda():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

# ---------- Analyzer (CUDA fast path with CPU fallback) ----------


def analyze_gpu_fast(path, roi=None, grace_frames=10,
                     scale=0.5, stride=5, refine_seconds=3.0,
                     diff_thresh=18, min_pixels_outside=1200):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 40.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Background & ROI (CPU)
    bg_full = build_background(path)
    if roi is None:
        roi = auto_detect_shelter(bg_full) or manual_roi_from_first_frame(path)

    # Inflate ROI slightly to avoid edge bleed
    x, y, w, h = roi
    pad = 3
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(W - x, w + 2*pad)
    h = min(H - y, h + 2*pad)
    roi = (x, y, w, h)

    # Scaled resources
    SW, SH = int(W*scale), int(H*scale)
    bg_s = cv2.resize(bg_full, (SW, SH), interpolation=cv2.INTER_AREA)
    shelter_mask = np.zeros((SH, SW), np.uint8)
    sx, sy = int(x*scale), int(y*scale)
    sw, sh = max(1, int(w*scale)), max(1, int(h*scale))
    shelter_mask[sy:sy+sh, sx:sx+sw] = 255
    outside_mask = cv2.bitwise_not(shelter_mask)

    use_cuda = has_cuda()

    if use_cuda:
        bg_gpu = cv2.cuda_GpuMat()
        bg_gpu.upload(bg_s)
        mask_gpu = cv2.cuda_GpuMat()
        mask_gpu.upload(outside_mask)
        K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_op = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_OPEN, cv2.CV_8UC1, K3)
    else:
        K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def outside_present_cpu(gray_s):
        diff = cv2.absdiff(gray_s, bg_s)
        _, th = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        th = cv2.bitwise_and(th, outside_mask)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, K3, iterations=1)
        return (th.sum() // 255) >= min_pixels_outside

    def outside_present_gpu(gray_s):
        g = cv2.cuda_GpuMat()
        g.upload(gray_s)
        d = cv2.cuda.absdiff(g, bg_gpu)
        _, th = cv2.cuda.threshold(d, diff_thresh, 255, cv2.THRESH_BINARY)
        th = cv2.cuda.bitwise_and(th, mask_gpu)
        th = morph_op.apply(th)
        m = cv2.cuda.mean(th)[0]  # 0..255
        count = int((m/255.0) * (SW*SH))
        return count >= min_pixels_outside

    outside_present = outside_present_gpu if use_cuda else outside_present_cpu

    # --- Coarse pass (stride) ---
    state_out = False
    missing = 0
    candidates = []  # refine windows (a,b)
    events_coarse = []

    f = 0
    while f < N:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        g = read_gray(fr)
        g = cv2.resize(g, (SW, SH), interpolation=cv2.INTER_AREA)

        present = outside_present(g)
        was = state_out
        if present:
            state_out = True
            missing = 0
        else:
            missing += stride
            if missing >= grace_frames:
                state_out = False

        if state_out != was:
            half = int(refine_seconds * fps)
            a = max(0, f - half)
            b = min(N-1, f + half)
            candidates.append((a, b))
            events_coarse.append((f, f/fps, "outside" if state_out else "inside",
                                  "exit_shelter" if state_out else "enter_shelter"))
        f += stride

    # --- Refine windows at full fps ---
    events = []
    segments = []
    seg_open = None

    for a, b in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        local_state = None
        local_missing = 0
        k = a
        while k <= b:
            ok, fr = cap.read()
            if not ok:
                break
            g = read_gray(fr)
            g = cv2.resize(g, (SW, SH), interpolation=cv2.INTER_AREA)
            present = outside_present(g)

            if present:
                new_state = True
                local_missing = 0
            else:
                local_missing += 1
                new_state = False if local_missing >= grace_frames else (
                    local_state if local_state is not None else False)

            if local_state is not None and new_state != local_state:
                events.append((k, k/fps, "outside" if new_state else "inside",
                               "exit_shelter" if new_state else "enter_shelter"))
                if new_state and seg_open is None:
                    seg_open = k
                elif (not new_state) and seg_open is not None:
                    segments.append((seg_open, k-1))
                    seg_open = None

            local_state = new_state
            k += 1

    if seg_open is not None:
        segments.append((seg_open, N-1))

    # Merge overlaps
    segments.sort()
    merged = []
    for s, e in segments:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    cap.release()
    return dict(fps=fps, roi=roi, events=sorted(events, key=lambda x: x[0]),
                segments=[tuple(x) for x in merged], size=(W, H), total_frames=N)

# ---------- Cutting with FFmpeg ----------


def cut_segments_ffmpeg(in_path, out_path, segments, fps, pad_s=0.0,
                        reencode=True, lossless=True):
    """
    Create per-segment clips in a temp dir, then concat to final file at out_path.
    By default uses lossless FFV1 gray for analysis-perfect output.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=str(out_path.parent)) as tmpdir:
        tmpdir = Path(tmpdir)
        part_paths = []
        for i, (a, b) in enumerate(segments):
            aa = max(0, a - int(pad_s*fps))
            bb = b + int(pad_s*fps)
            start = aa / fps
            dur = (bb - aa + 1) / fps
            part = tmpdir / \
                f"clip_{i:04d}.mkv" if lossless else tmpdir / \
                f"clip_{i:04d}.mp4"
            part_paths.append(part)

            if reencode:
                if lossless:
                    cmd = ["ffmpeg", "-loglevel", "error", "-y",
                           "-ss", f"{start:.3f}", "-i", str(in_path),
                           "-t", f"{dur:.3f}",
                           "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "gray",
                           "-an", str(part)]
                else:
                    cmd = ["ffmpeg", "-loglevel", "error", "-y",
                           "-ss", f"{start:.3f}", "-i", str(in_path),
                           "-t", f"{dur:.3f}",
                           "-vf", "format=gray", "-c:v", "libx264", "-crf", "16", "-preset", "veryfast",
                           "-pix_fmt", "yuv420p", "-an", str(part)]
            else:
                cmd = ["ffmpeg", "-loglevel", "error", "-y",
                       "-ss", f"{start:.3f}", "-i", str(in_path),
                       "-t", f"{dur:.3f}",
                       "-c", "copy", "-an", str(part)]
            subprocess.run(cmd, check=True)

        list_file = tmpdir / "concat_list.txt"
        with list_file.open("w") as f:
            for p in part_paths:
                f.write(f"file '{p.as_posix()}'\n")

        # Concat
        if reencode and lossless:
            final = out_path.with_suffix(".mkv")
            cmd = ["ffmpeg", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", str(list_file),
                   "-c", "copy", "-an", str(final)]
        elif reencode and not lossless:
            final = out_path.with_suffix(".mp4")
            cmd = ["ffmpeg", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", str(list_file),
                   "-c", "copy", "-an", str(final)]
        else:
            # stream copy concat
            final = out_path.with_suffix(".mp4")
            cmd = ["ffmpeg", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", str(list_file),
                   "-c", "copy", "-an", str(final)]
        subprocess.run(cmd, check=True)
        return str(final)


def write_events_csv(events, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "state", "transition"])
        for fr, t, st, tr in events:
            w.writerow([fr, f"{t:.3f}", st, tr])

# ---------- Main ----------


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Cut exit→entry segments using CUDA-accelerated detection (fallback to CPU).")
    ap.add_argument(
        "input", help="input video path (Y800/gray/MJPG/FFV1, etc.)")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("x", "y", "w", "h"),
                    help="manual shelter ROI (x y w h). If omitted: auto-detect → manual fallback.")
    ap.add_argument("--grace", type=int, default=10,
                    help="frames to wait before declaring inside")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="downscale factor for detection (0.5–0.75 good)")
    ap.add_argument("--stride", type=int, default=7,
                    help="coarse pass frame stride (e.g., 5–10)")
    ap.add_argument("--refine", type=float, default=3.0,
                    help="seconds around coarse transitions to refine at full fps")
    ap.add_argument("--diff", type=int, default=18,
                    help="absdiff threshold (pixel) for foreground")
    ap.add_argument("--min-pixels", type=int, default=1200,
                    help="minimum foreground pixels outside shelter")
    ap.add_argument("--pad", type=float, default=0.1,
                    help="seconds to pad around kept segments")
    ap.add_argument("--lossless", action="store_true",
                    help="use lossless FFV1 gray for segments/final (recommended)")
    ap.add_argument("--reencode", action="store_true",
                    help="re-encode segments for exact cuts (recommended)")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    # Where to save outputs: same directory as input
    out_base = in_path.with_suffix("").name + "_exit_to_entry"
    out_dir = in_path.parent
    events_csv = out_dir / (in_path.with_suffix("").name + "_events.csv")
    roi_json = out_dir / (in_path.with_suffix("").name + "_roi.json")
    out_final = out_dir / out_base  # suffix decided in cut function

    roi = tuple(args.roi) if args.roi else None

    res = analyze_gpu_fast(
        in_path, roi=roi, grace_frames=args.grace,
        scale=args.scale, stride=args.stride, refine_seconds=args.refine,
        diff_thresh=args.diff, min_pixels_outside=args.min_pixels
    )

    # Persist ROI & events next to the video
    with roi_json.open("w") as f:
        json.dump({"roi": res["roi"], "fps": res["fps"],
                  "size": res["size"]}, f, indent=2)
    write_events_csv(res["events"], str(events_csv))

    print(f"Detected ROI (x,y,w,h): {res['roi']}")
    print(f"Exit→entry segments: {len(res['segments'])}")

    if len(res["segments"]) == 0:
        print("No outside segments found; nothing to cut.")
        sys.exit(0)

    # Cut and save final next to input
    final_path = cut_segments_ffmpeg(
        in_path, out_final, res["segments"], res["fps"],
        pad_s=args.pad, reencode=(args.reencode or True), lossless=(args.lossless or True)
    )
    print(f"Events CSV: {events_csv}")
    print(f"ROI JSON:   {roi_json}")
    print(f"Final video: {final_path}")


if __name__ == "__main__":
    main()
