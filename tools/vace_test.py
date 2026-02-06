import os
import re
import cv2
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

FOLDER_RE = re.compile(r"^sample_(\d{4})_clip(\d{2})$")

VIDEO_MAP = {
    "generated": "generated.mp4",
    "gt": "clip_video.mp4",
    "handpose": "clip_vace_video_mask.mp4",
}

OUT_SUFFIX = {
    "generated": "generated.mp4",
    "gt": "gt.mp4",
    "handpose": "handpose.mp4",
}


def _list_sample_clips(root_dir: str) -> Dict[str, Dict[int, str]]:
    groups = defaultdict(dict)
    for name in os.listdir(root_dir):
        p = os.path.join(root_dir, name)
        if not os.path.isdir(p):
            continue
        m = FOLDER_RE.match(name)
        if not m:
            continue
        sid, cidx = m.group(1), int(m.group(2))
        groups[sid][cidx] = p
    return dict(groups)


def _open_video_info(video_path: str) -> Tuple[cv2.VideoCapture, float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"Cannot read frame to get shape: {video_path}")
        h, w = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return cap, float(fps), w, h


def _concat_videos(
    in_paths: List[str],
    out_path: str,
    drop_first_frame_for_rest: bool = True,
    resize_to_first: bool = True,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cap0, fps, W, H = _open_video_info(in_paths[0])
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_path}")

    for vid_i, vp in enumerate(in_paths):
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            writer.release()
            raise RuntimeError(f"Cannot open video: {vp}")

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if drop_first_frame_for_rest and vid_i > 0 and frame_idx == 0:
                frame_idx += 1
                continue
            if resize_to_first and (frame.shape[1] != W or frame.shape[0] != H):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
            frame_idx += 1
        cap.release()

    writer.release()


def merge_all_samples(root_dir: str, out_dir: Optional[str] = None, sample_min: int = 0, sample_max: int = 9999):
    if out_dir is None:
        out_dir = root_dir

    groups = _list_sample_clips(root_dir)
    allowed_ids = [f"{i:04d}" for i in range(sample_min, sample_max + 1)]
    groups = {sid: clips for sid, clips in groups.items() if sid in allowed_ids}

    if not groups:
        raise RuntimeError(f"No folders matched pattern in: {root_dir}")

    for sid, clip_map in sorted(groups.items(), key=lambda x: x[0]):
        clips = sorted(clip_map.keys())
        if len(clips) == 0:
            continue

        print(f"\n[Sample {sid}] clips: {clips}")

        for key, fname in VIDEO_MAP.items():
            in_paths = []
            for c in clips:
                fpath = os.path.join(clip_map[c], fname)
                if not os.path.isfile(fpath):
                    print(f"  - WARN missing {fname} in clip{c:02d}: {fpath} (skip)")
                    continue
                in_paths.append(fpath)

            if len(in_paths) == 0:
                print(f"  - SKIP {key}: no input videos found")
                continue

            out_path = os.path.join(out_dir, f"sample_{sid}_{OUT_SUFFIX[key]}")
            print(f"  - Merge {key}: {len(in_paths)} videos -> {out_path}")
            _concat_videos(in_paths, out_path, drop_first_frame_for_rest=True, resize_to_first=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root dir containing sample_xxxx_clipyy folders")
    parser.add_argument("--out", default=None, help="Output dir (default: same as --root)")
    parser.add_argument("--sample-min", type=int, default=0, help="Minimum sample id (default: 0)")
    parser.add_argument("--sample-max", type=int, default=9999, help="Maximum sample id (default: 9999)")
    args = parser.parse_args()

    merge_all_samples(args.root, args.out, args.sample_min, args.sample_max)


# 合并 sample_0000 到 sample_0016 的所有片段
# python /root/paddlejob/workspace/huangxuan/DiffSynth-Studio-New/tools/vace_test.py \
#   --root /root/paddlejob/workspace/huangxuan/DiffSynth-Studio-New/0112/vace_swap_multigpu_s1100_long_ref_first \
#   --sample-min 0 \
#   --sample-max 15
