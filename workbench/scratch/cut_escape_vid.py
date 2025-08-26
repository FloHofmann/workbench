import cv2, numpy as np, subprocess, csv, os, json
from pathlib import Path

def read_gray(frame):
    # Robustly get a single-channel grayscale ndarray from any input frame shape
    if len(frame.shape) == 2:
        return frame
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def build_background(path, sample_every=300, max_samples=50):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, min(sample_every, max(1, total // max_samples)))
    idx = 0
    while idx < total and len(frames) < max_samples:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok: break
        frames.append(read_gray(f))
        idx += step
    cap.release()
    if not frames:
        raise RuntimeError("Could not sample frames for background.")
    return np.median(np.stack(frames, 0), 0).astype(np.uint8)

def auto_detect_shelter(gray_bg, min_rect_area_frac=0.02):
    inv = cv2.bitwise_not(gray_bg)
    th = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8), 2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray_bg.shape
    min_area = min_rect_area_frac * (H*W)
    best, best_area = None, 0
    for c in cnts:
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        area = cv2.contourArea(approx)
        if area > best_area and area > min_area and len(approx) in (4,5):
            best, best_area = approx, area
    return cv2.boundingRect(best) if best is not None else None

def manual_roi_from_first_frame(path):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    ok, f = cap.read()
    cap.release()
    if not ok: raise RuntimeError("Cannot read first frame for manual ROI")
    r = cv2.selectROI("Select shelter ROI (ENTER to confirm)", f, False, True)
    cv2.destroyWindow("Select shelter ROI (ENTER to confirm)")
    x,y,w,h = map(int, r)
    return (x,y,w,h)

def make_mask(shape, roi):
    mask = np.zeros(shape, np.uint8)
    x,y,w,h = roi
    mask[y:y+h, x:x+w] = 255
    return mask

def analyze(path, roi=None, min_mouse_area=600, grace_frames=10):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 40.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bg = build_background(path)
    if roi is None:
        roi = auto_detect_shelter(bg) or manual_roi_from_first_frame(path)

    # Slightly inflate ROI to avoid counting edge pixels as "outside"
    ix,iy,iw,ih = roi
    pad = 3
    ix = max(0, ix - pad); iy = max(0, iy - pad)
    iw = min(W - ix, iw + 2*pad); ih = min(H - iy, ih + 2*pad)
    roi = (ix,iy,iw,ih)

    shelter_mask = make_mask((H,W), roi)
    outside_mask = cv2.bitwise_not(shelter_mask)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=False)

    # State machine
    #   inside = not seeing a blob outside for >= grace_frames
    state_outside = False
    missing = 0

    events = []        # (frame, time_s, state, transition)
    segments = []      # (exit_frame, entry_frame) — what we will keep
    seg_open = None    # frame index where we exited (mouse appears outside)

    k = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        gray = read_gray(frame)
        masked = cv2.bitwise_and(gray, outside_mask)

        fg = fgbg.apply(masked)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 2)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = max((cv2.contourArea(c) for c in cnts), default=0)

        was_outside = state_outside
        if biggest > min_mouse_area:
            state_outside = True
            missing = 0
        else:
            missing += 1
            if missing >= grace_frames:
                state_outside = False

        if state_outside != was_outside:
            trans = "exit_shelter" if state_outside else "enter_shelter"
            events.append((k, k/fps, "outside" if state_outside else "inside", trans))
            if state_outside:
                seg_open = k              # start exit→entry window
            else:
                if seg_open is not None:
                    segments.append((seg_open, k-1))
                    seg_open = None

        k += 1

    # If video ends while outside, close the last segment to end
    if seg_open is not None:
        segments.append((seg_open, total-1))

    cap.release()
    return dict(fps=fps, roi=roi, events=events, segments=segments,
                size=(W,H), total_frames=total)

def write_events_csv(events, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame","time_s","state","transition"])
        for fr,t,st,tr in events:
            w.writerow([fr, f"{t:.3f}", st, tr])

def cut_segments_ffmpeg(in_path, segments, fps, pad_s=0.0, reencode=True, lossless=True):
    # Create per-segment clips, then concat. For analysis, prefer lossless gray (FFV1).
    out_paths = []
    for i,(a,b) in enumerate(segments):
        aa = max(0, a - int(pad_s*fps))
        bb = b + int(pad_s*fps)
        start = aa / fps
        dur   = (bb - aa + 1) / fps
        out_p = f"clip_{i:04d}.mkv" if lossless else f"clip_{i:04d}.mp4"
        out_paths.append(out_p)

        if reencode:
            if lossless:
                # FFV1 lossless, grayscale, intra-friendly (perfect for tracking)
                cmd = ["ffmpeg","-y","-ss",f"{start:.3f}","-i",in_path,"-t",f"{dur:.3f}",
                       "-c:v","ffv1","-level","3","-g","1","-pix_fmt","gray",
                       "-an", out_p]
            else:
                # Near-lossless H.264, grayscale
                cmd = ["ffmpeg","-y","-ss",f"{start:.3f}","-i",in_path,"-t",f"{dur:.3f}",
                       "-vf","format=gray","-c:v","libx264","-crf","16","-preset","veryfast",
                       "-pix_fmt","yuv420p","-an", out_p]
        else:
            # Stream copy (only safe if your container/keyframes allow accurate cuts)
            cmd = ["ffmpeg","-y","-ss",f"{start:.3f}","-i",in_path,"-t",f"{dur:.3f}",
                   "-c","copy","-an", out_p]
        subprocess.run(cmd, check=True)

    # Concat all clips into one video
    with open("concat_list.txt","w") as f:
        for p in out_paths:
            f.write(f"file '{p}'\n")
    final_out = "mouse_exit_to_entry.mkv" if (reencode and lossless) else "mouse_exit_to_entry.mp4"
    concat_cmd = ["ffmpeg","-y","-f","concat","-safe","0","-i","concat_list.txt",
                  "-c","copy" if (reencode and lossless) else ("libx264" if reencode else "copy"),
                  "-an", final_out]
    subprocess.run(concat_cmd, check=True)
    return final_out

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="input video (Y800/gray)")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("x","y","w","h"),
                    help="manual shelter ROI (x y w h). If omitted: auto → manual GUI fallback.")
    ap.add_argument("--min-area", type=int, default=600, help="min contour area for mouse blob")
    ap.add_argument("--grace", type=int, default=10, help="frames to wait before declaring inside")
    ap.add_argument("--pad", type=float, default=0.0, help="seconds to pad around exit→entry clips")
    ap.add_argument("--reencode", action="store_true", help="re-encode segments (recommended for exact frame cuts)")
    ap.add_argument("--lossless", action="store_true", help="use lossless FFV1 gray for segments+final")
    ap.add_argument("--save-roi", default="roi.json", help="write detected ROI for reuse")
    ap.add_argument("--csv", default="events.csv", help="events CSV")
    args = ap.parse_args()

    roi = tuple(args.roi) if args.roi else None
    res = analyze(args.input, roi=roi, min_mouse_area=args.min_area, grace_frames=args.grace)

    # persist ROI for later runs
    with open(args.save_roi, "w") as f: json.dump({"roi": res["roi"]}, f)
    print("ROI:", res["roi"])
    print("segments:", len(res["segments"]))
    write_events_csv(res["events"], args.csv)
    print("Wrote", args.csv)

    final = cut_segments_ffmpeg(args.input, res["segments"], res["fps"],
                                pad_s=args.pad,
                                reencode=args.reencode or True,
                                lossless=args.lossless or True)
    print("Final video:", final)

