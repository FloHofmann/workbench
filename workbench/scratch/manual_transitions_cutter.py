#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
import subprocess
from pathlib import Path
import cv2
import numpy as np

HELP_TEXT = [
    "Controls:",
    "  SPACE: play/pause     , / . : -1 / +1 frame",
    "  z / x : -1s / +1s     c / v : -10s / +10s",
    "  o     : mark EXIT (outside starts)",
    "  i     : mark ENTER (outside ends)",
    "  u     : undo last mark",
    "  s     : save marks CSV",
    "  ENTER : finish & cut",
    "  q     : quit without cutting",
    "Scrub with the trackbar at the bottom.",
]


def draw_hud(img, text_lines, pos=(10, 10)):
    # draw semi-transparent box with text
    out = img.copy()
    h = 18 * (len(text_lines) + 1)
    w = 560
    overlay = out.copy()
    cv2.rectangle(overlay, pos, (pos[0]+w, pos[1]+h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
    y = pos[1] + 18
    for line in text_lines:
        cv2.putText(out, line, (pos[0]+8, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18
    return out


def format_time(frame, fps):
    t = frame / max(fps, 1e-6)
    h = int(t // 3600)
    t -= 3600*h
    m = int(t // 60)
    t -= 60*m
    s = t
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def ensure_bgr(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 1:
        return cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    return img


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def write_events_csv(path, events):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "label"])
        for fr, t, lab in events:
            w.writerow([fr, f"{t:.3f}", lab])


def compute_segments_from_events(events, fps):
    """
    Expect alternating EXIT (outside starts) and ENTER (outside ends).
    Return list of (start_frame, end_frame) for outside windows.
    Ignore unmatched trailing EXIT.
    If events start with ENTER, it’s ignored.
    """
    segments = []
    open_start = None
    for fr, t, lab in sorted(events, key=lambda x: x[0]):
        if lab == "exit_shelter":
            if open_start is None:
                open_start = fr
            else:
                # double-exit: close previous at fr-1 and open new
                segments.append((open_start, fr-1))
                open_start = fr
        elif lab == "enter_shelter":
            if open_start is not None:
                segments.append((open_start, max(fr-1, open_start)))
                open_start = None
            else:
                # ENTER without open: ignore
                pass
    return segments


def cut_with_ffmpeg(in_path, out_path_base, segments, fps, pad_s=0.0, lossless=True, reencode=True):
    out_dir = Path(in_path).parent
    out_base = Path(out_path_base)
    ext = ".mkv" if lossless else ".mp4"
    final_out = (out_dir / f"{out_base.name}{ext}").resolve()

    import tempfile
    with tempfile.TemporaryDirectory(dir=str(out_dir)) as tmpd:
        tmpd = Path(tmpd)
        parts = []
        for i, (a, b) in enumerate(segments):
            aa = max(0, a - int(pad_s*fps))
            bb = b + int(pad_s*fps)
            start = aa / fps
            dur = (bb - aa + 1) / fps
            part = tmpd / f"clip_{i:04d}{ext}"
            parts.append(part)
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

        list_file = tmpd / "concat_list.txt"
        with list_file.open("w") as f:
            for p in parts:
                f.write(f"file '{p.as_posix()}'\n")

        # concat
        cmd = ["ffmpeg", "-loglevel", "error", "-y",
               "-f", "concat", "-safe", "0", "-i", str(list_file),
               "-c", "copy", "-an", str(final_out)]
        subprocess.run(cmd, check=True)

    return str(final_out)


def main():
    ap = argparse.ArgumentParser(
        description="Manually mark exit/enter transitions and cut outside-only segments.")
    ap.add_argument("input", help="video path")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="seconds of padding around each kept segment")
    ap.add_argument("--lossless", action="store_true",
                    help="use lossless FFV1 gray (mkv) for output")
    ap.add_argument("--copy", action="store_true",
                    help="attempt stream copy for cuts (only accurate if intra-only)")
    ap.add_argument("--fps", type=float, default=None,
                    help="override FPS if the file lacks it")
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(str(in_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Could not open video.")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = args.fps or (cap.get(cv2.CAP_PROP_FPS) or 40.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    window = "manual cutter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(window, min(1280, max(640, w)), min(960, max(480, h+120)))

    # Trackbar to scrub frames
    current_frame = 0
    updating_from_trackbar = False

    def on_trackbar(val):
        nonlocal current_frame, updating_from_trackbar
        updating_from_trackbar = True
        current_frame = int(val)
        updating_from_trackbar = False

    cv2.createTrackbar("frame", window, 0, max(total-1, 1), on_trackbar)

    # Playback state
    playing = False
    last_time = time.time()
    speed_mult = 1.0

    # User marks: list of tuples (frame, time_s, label)
    # labels: "exit_shelter" (outside starts), "enter_shelter" (outside ends)
    events = []

    # Utility: overlay timeline bar and info
    def render_frame(idx):
        frame = read_frame(cap, idx)
        if frame is None:
            return None
        bgr = ensure_bgr(frame)

        # draw a thin timeline at bottom
        tl_h = 20
        out = np.zeros((bgr.shape[0] + tl_h + 100,
                       bgr.shape[1], 3), dtype=np.uint8)
        out[:bgr.shape[0], :, :] = bgr

        # timeline bar background
        cv2.rectangle(
            out, (0, bgr.shape[0]), (bgr.shape[1], bgr.shape[0]+tl_h), (40, 40, 40), -1)
        x_pos = int((idx / max(total-1, 1)) * (bgr.shape[1]-1))
        cv2.line(out, (x_pos, bgr.shape[0]), (x_pos,
                 bgr.shape[0]+tl_h), (200, 200, 200), 2)

        # mark event ticks
        for fr, t, lab in events:
            tx = int((fr / max(total-1, 1)) * (bgr.shape[1]-1))
            color = (0, 255, 0) if lab == "exit_shelter" else (
                0, 165, 255)  # green exit, orange enter
            cv2.circle(out, (tx, bgr.shape[0]+tl_h//2), 3, color, -1)

        # header info
        info = [
            f"Frame {idx+1}/{total}  |  {format_time(idx, fps)}  |  FPS: {fps:.3f}  |  {
                'PLAY' if playing else 'PAUSE'} x{speed_mult}",
            f"Marks: {len(events)}   Next suggested: {'EXIT(o)' if (
                len(events) == 0 or events[-1][2] == 'enter_shelter') else 'ENTER(i)'}"
        ]
        out = draw_hud(out, HELP_TEXT + [""] + info, pos=(10, 10))
        return out

    while True:
        # auto-advance
        if playing:
            now = time.time()
            dt = now - last_time
            step = int(dt * fps * speed_mult)
            if step >= 1:
                current_frame = min(total-1, current_frame + step)
                last_time = now

        # clamp & show
        current_frame = max(0, min(total-1, current_frame))
        if not updating_from_trackbar:
            cv2.setTrackbarPos("frame", window, current_frame)

        vis = render_frame(current_frame)
        if vis is None:
            break
        cv2.imshow(window, vis)

        # wait a bit
        key = cv2.waitKey(1) & 0xFF

        if key == 255:  # no key
            continue
        elif key in (ord('q'), 27):  # q or ESC
            print("Exiting without cutting.")
            break
        elif key == 13:  # ENTER → finish & cut
            break_and_cut = True
            # fall through to after loop
            playstate = False
            playing = False
            # exit loop to cut
            finalize = True
            # consume key then break loop
            finalize = True
            # ensure we exit loop
            break
        elif key == ord(' '):
            playing = not playing
            last_time = time.time()
        elif key == ord(','):
            playing = False
            current_frame = max(0, current_frame - 1)
        elif key == ord('.'):
            playing = False
            current_frame = min(total-1, current_frame + 1)
        elif key == ord('z'):
            playing = False
            current_frame = max(0, current_frame - int(fps))
        elif key == ord('x'):
            playing = False
            current_frame = min(total-1, current_frame + int(fps))
        elif key == ord('c'):
            playing = False
            current_frame = max(0, current_frame - int(10*fps))
        elif key == ord('v'):
            playing = False
            current_frame = min(total-1, current_frame + int(10*fps))
        elif key == ord('o'):  # EXIT (outside starts)
            t = current_frame / fps
            events.append((current_frame, t, "exit_shelter"))
            print(f"[mark] EXIT at frame {current_frame} ({
                  format_time(current_frame, fps)})")
        elif key == ord('i'):  # ENTER (outside ends)
            t = current_frame / fps
            events.append((current_frame, t, "enter_shelter"))
            print(f"[mark] ENTER at frame {
                  current_frame} ({format_time(current_frame, fps)})")
        elif key == ord('u'):
            if events:
                fr, t, lab = events.pop()
                print(f"[undo] removed {lab} at frame {
                      fr} ({format_time(fr, fps)})")
        elif key == ord('s'):
            out_csv = in_path.parent / \
                f"{in_path.with_suffix('').name}_manual_events.csv"
            write_events_csv(str(out_csv), events)
            print(f"[save] events → {out_csv}")
        elif key == ord('1'):
            speed_mult = 1.0
            print("Playback speed: 1.0×")
        elif key == ord('2'):
            speed_mult = 1.25
            print("Playback speed: 1.25×")
        elif key == ord('3'):
            speed_mult = 1.5
            print("Playback speed: 1.5×")
        elif key == ord('4'):
            speed_mult = 2.0
            print("Playback speed: 2.0×")
        elif key == ord('5'):
            speed_mult = 3.0
            print("Playback speed: 3.0×")
        elif key == ord('6'):
            speed_mult = 4.0
            print("Playback speed: 4.0×")

    cv2.destroyAllWindows()
    cap.release()

    # If ENTER was pressed, finalize (cut)
    if 'finalize' in locals() and finalize:
        if not events:
            print("No marks placed; nothing to cut.")
            return
        # Save events CSV
        out_csv = in_path.parent / \
            f"{in_path.with_suffix('').name}_manual_events.csv"
        write_events_csv(str(out_csv), events)
        print(f"Events saved: {out_csv}")

        # Build segments from exit→enter pairs
        segments = compute_segments_from_events(events, fps)
        if not segments:
            print("No valid EXIT→ENTER pairs found; nothing to cut.")
            return

        # Final output base path (same folder)
        out_base = f"{in_path.with_suffix('').name}_manual_exit_to_entry"
        try:
            final_path = cut_with_ffmpeg(
                str(in_path),
                out_base,
                segments,
                fps,
                pad_s=args.pad,
                lossless=args.lossless or True,     # default lossless on
                reencode=(not args.copy) or True    # default reencode on
            )
        except FileNotFoundError as e:
            print(
                "ffmpeg not found. Please install ffmpeg and ensure it is in your PATH.")
            print("Get it from https://ffmpeg.org/")
            sys.exit(1)
        print(f"Final video: {final_path}")
        print("Done.")


if __name__ == "__main__":
    main()
