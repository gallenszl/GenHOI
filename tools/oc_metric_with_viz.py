# oc_metric_with_viz.py
import os
import math
import numpy as np
from typing import Optional, Tuple, List, Dict

from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F

# ==== 后端可选依赖：open-clip 优先，其次 OpenAI clip ====
_CLIP_BACKEND = None
try:
    import open_clip
    _CLIP_BACKEND = "open_clip"
except Exception:
    try:
        import clip as openai_clip
        _CLIP_BACKEND = "clip"
    except Exception:
        _CLIP_BACKEND = None

# ==== 读视频：优先 decord，其次 OpenCV ====
_HAS_DECORD = False
try:
    import decord
    _HAS_DECORD = True
    decord.bridge.set_bridge("native")
except Exception:
    pass

_HAS_CV2 = False
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    pass

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".gif"}


def _is_video_path(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in _VIDEO_EXTS


def _load_single_frame_from_video_as_pil(video_path: str) -> Image.Image:
    """读取仅1帧的视频为 PIL.Image（RGB）。若是多帧视频则报错。"""
    if _HAS_DECORD:
        vr = decord.VideoReader(video_path)
        n = len(vr)
        if n < 1:
            raise RuntimeError(f"ref video 为空: {video_path}")
        if n > 1:
            raise ValueError(f"ref video 有 {n} 帧，但期望 1 帧: {video_path}")
        arr = vr.get_batch([0]).asnumpy()[0]  # RGB
        return Image.fromarray(arr).convert("RGB")
    elif _HAS_CV2:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV 打开失败: {video_path}")
        ok, frame0 = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"无法读取首帧: {video_path}")
        ok2, _ = cap.read()
        cap.release()
        if ok2:
            raise ValueError(f"ref video 超过 1 帧: {video_path}")
        frame_rgb = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb).convert("RGB")
    else:
        raise RuntimeError("需要 decord 或 opencv-python 才能读取视频。")


def _load_ref_image_or_single_frame(ref_path: str) -> Image.Image:
    """ref 可以是图片或仅1帧视频，返回 PIL.Image(RGB)。"""
    if _is_video_path(ref_path):
        return _load_single_frame_from_video_as_pil(ref_path)
    return Image.open(ref_path).convert("RGB")


def _read_video_rgb(path: str, stride: int = 1) -> Tuple[List[np.ndarray], float]:
    """
    读取视频为 list[H,W,3] (RGB, uint8)；返回 (frames, fps)
    stride>1 会跳帧。
    """
    if _HAS_DECORD:
        vr = decord.VideoReader(path)
        fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 25.0
        idx = list(range(0, len(vr), stride))
        if not idx:
            return [], fps
        arr = vr.get_batch(idx).asnumpy()  # (N,H,W,3) RGB uint8
        return [a for a in arr], fps
    elif _HAS_CV2:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
        return frames, float(fps)
    else:
        raise RuntimeError("需要 decord 或 opencv-python 读取视频。")


def _read_mask_video_bool(path: str, stride: int = 1, thr: int = 127) -> List[np.ndarray]:
    """
    读取 0/1 bbox mask 视频，返回 list[H,W] (bool)。
    """
    if _HAS_DECORD:
        vr = decord.VideoReader(path)
        idx = list(range(0, len(vr), stride))
        if not idx:
            return []
        arr = vr.get_batch(idx).asnumpy()  # (N,H,W,3) 或 (N,H,W,1)
        if arr.ndim == 4 and arr.shape[-1] > 1:
            gray = np.round(arr[..., :3].mean(-1)).astype(np.uint8)
        else:
            gray = arr.squeeze(-1).astype(np.uint8)
        return [g > thr for g in gray]
    elif _HAS_CV2:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV 无法打开视频: {path}")
        masks = []
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                masks.append(gray > thr)
            i += 1
        cap.release()
        return masks
    else:
        raise RuntimeError("需要 decord 或 opencv-python 读取视频。")


def _bbox_from_mask(m: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """从二值 mask(HxW) 得到 (x0,y0,x1,y1)；若全空返回 None。"""
    ys, xs = np.where(m)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return (x0, y0, x1, y1)


def _resize_pad_to_square(img: Image.Image, size: int) -> Image.Image:
    """等比缩放并居中填充到 size×size。"""
    w, h = img.size
    s = size
    scale = min(s / w, s / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img2 = img.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (s, s), (0, 0, 0))
    canvas.paste(img2, ((s - nw) // 2, (s - nh) // 2))
    return canvas


def _load_clip(device: str = "cuda"):
    if _CLIP_BACKEND == "open_clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        model.eval()
        return ("open_clip", model, preprocess)
    elif _CLIP_BACKEND == "clip":
        model, preprocess = openai_clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        return ("clip", model, preprocess)
    else:
        raise RuntimeError(
            "未检测到 CLIP。请先安装 open-clip-torch 或 clip：\n"
            "pip install open-clip-torch\n"
            "或 pip install git+https://github.com/openai/CLIP.git"
        )


@torch.no_grad()
def _encode_pil(img: Image.Image, clip_backend, model, preprocess, device) -> torch.Tensor:
    if clip_backend == "open_clip":
        # open-clip 的 preprocess 已经是 torchvision 变换到 224
        t = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(t)
    else:
        # OpenAI clip
        t = preprocess(img).unsqueeze(0).to(device)
        feat = model.encode_image(t)
    feat = F.normalize(feat, dim=-1)
    return feat  # (1, D)


def _draw_bbox_and_score(
    frame_rgb: np.ndarray,
    bbox: Tuple[int, int, int, int],
    score: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    在 RGB 帧上画框和分数，返回新帧（RGB）。
    """
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    x0, y0, x1, y1 = bbox
    draw.rectangle([(x0, y0), (x1, y1)], outline=tuple(color), width=thickness)
    text = f"OC {score:.3f}"
    # 尝试加载等宽字体失败就用默认
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size=18)
    except Exception:
        font = ImageFont.load_default()
    tw, th = draw.textlength(text, font=font), 18
    # 文本背景
    draw.rectangle([(x0, max(0, y0 - th - 4)), (x0 + tw + 8, y0)], fill=(0, 0, 0))
    draw.text((x0 + 4, max(0, y0 - th - 2)), text, fill=(255, 255, 255), font=font)
    return np.array(img)


@torch.no_grad()
def compute_oc_metric_with_viz(
    video_path: str,
    bbox_mask_video_path: str,
    ref_image_path: Optional[str] = None,    # 旧接口：单张图/单帧视频
    ref_mask_path: Optional[str] = None,     # 可选：对 ref 图也裁一个 mask/bbox
    output_vis_path: Optional[str] = None,   # 可选：输出带 bbox+分数 的视频
    stride: int = 1,                         # 抽帧间隔
    device: Optional[str] = None,
    aggregate: str = "mean",                 # "mean" | "median"
    # ---- 新增：GT 参考视频（与 generated 逐帧比较）----
    ref_video_path: Optional[str] = None,
) -> Dict:
    """
    OC (Object-CLIP) 指标：
      - 若提供 ref_video_path：逐帧在 bbox 内比较 Generated vs GT；
      - 否则：在 bbox 内比较 Generated vs 单张 ref 图（兼容旧逻辑）。
    返回:
      {
        "overall": float,         # 聚合分数
        "per_frame": List[float], # 每帧分数，mask 空帧为 np.nan
        "n_frames": int,
        "n_valid": int,
        "vis_path": Optional[str]
      }
    """
    # 基本检查
    assert os.path.exists(video_path), f"video 不存在: {video_path}"
    assert os.path.exists(bbox_mask_video_path), f"bbox mask 视频不存在: {bbox_mask_video_path}"
    if (ref_image_path is None) and (ref_video_path is None):
        raise ValueError("必须提供 ref_image_path（旧模式）或 ref_video_path（逐帧 GT 模式）之一。")
    if (ref_image_path is not None) and (ref_video_path is not None):
        raise ValueError("ref_image_path 与 ref_video_path 只能二选一。")

    # 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 加载 CLIP
    backend, clip_model, preprocess = _load_clip(device)

    # 2) 读 generated 视频与 mask
    frames_rgb, fps = _read_video_rgb(video_path, stride=stride)  # list[H,W,3] RGB
    masks = _read_mask_video_bool(bbox_mask_video_path, stride=stride)  # list[H,W] bool
    if len(frames_rgb) == 0:
        raise RuntimeError("视频帧数为 0。")
    if len(masks) == 0:
        raise RuntimeError("mask 帧数为 0。")

    # 3) 参考端：两种模式
    use_video_ref = (ref_video_path is not None)

    if use_video_ref:
        # 逐帧 GT 模式
        assert os.path.exists(ref_video_path), f"GT 视频不存在: {ref_video_path}"
        gt_frames_rgb, _ = _read_video_rgb(ref_video_path, stride=stride)  # 与 generated 同 stride
        if len(gt_frames_rgb) == 0:
            raise RuntimeError("GT 视频帧数为 0。")

        # 对齐最短长度
        n = min(len(frames_rgb), len(masks), len(gt_frames_rgb))
        frames_rgb = frames_rgb[:n]
        masks = masks[:n]
        gt_frames_rgb = gt_frames_rgb[:n]

        ref_feat = None  # 不用单一特征
    else:
        # 旧逻辑：单张 ref 图或 1 帧视频
        assert ref_image_path is not None and os.path.exists(ref_image_path), f"ref 不存在: {ref_image_path}"
        n = min(len(frames_rgb), len(masks))
        frames_rgb = frames_rgb[:n]
        masks = masks[:n]

        # 参考图 & 可选参考 mask
        ref_img = _load_ref_image_or_single_frame(ref_image_path)
        if ref_mask_path is not None:
            # 参考 mask 既可图片，也可 1 帧视频
            if _is_video_path(ref_mask_path):
                ref_m_list = _read_mask_video_bool(ref_mask_path, stride=1)
                if len(ref_m_list) != 1:
                    raise ValueError(f"ref mask video 帧数={len(ref_m_list)}，期望=1: {ref_mask_path}")
                ref_mask = ref_m_list[0]
            else:
                m = Image.open(ref_mask_path).convert("L")
                ref_mask = (np.array(m) > 127)
            rb = _bbox_from_mask(ref_mask)
            if rb is not None:
                x0, y0, x1, y1 = rb
                ref_img_cropped = ref_img.crop((x0, y0, x1 + 1, y1 + 1))
            else:
                ref_img_cropped = ref_img
        else:
            ref_img_cropped = ref_img

        ref_feat = _encode_pil(ref_img_cropped, backend, clip_model, preprocess, device)  # (1,D)
        gt_frames_rgb = None  # 不用

    # 4) 可视化 writer
    vis_writer = None
    has_cv2_writer = False
    if output_vis_path is not None:
        if not _HAS_CV2:
            print("[warn] 未安装 opencv-python，无法保存可视化视频。")
            output_vis_path = None
        else:
            os.makedirs(os.path.dirname(output_vis_path) or ".", exist_ok=True)
            H, W, _ = frames_rgb[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            vis_writer = cv2.VideoWriter(output_vis_path, fourcc, fps, (W, H))
            has_cv2_writer = vis_writer.isOpened()
            if not has_cv2_writer:
                print("[warn] VideoWriter 打开失败，跳过可视化保存。")
                output_vis_path = None

    # 5) 逐帧在 bbox 内计算相似度
    per_scores: List[float] = []
    for i in range(n):
        fr = frames_rgb[i]
        m = masks[i]
        bbox = _bbox_from_mask(m)
        if bbox is None:
            per_scores.append(float("nan"))
            if output_vis_path is not None and has_cv2_writer:
                vis_writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            continue

        x0, y0, x1, y1 = bbox
        x1, y1 = min(x1, fr.shape[1] - 1), min(y1, fr.shape[0] - 1)
        gen_crop = Image.fromarray(fr[y0:y1 + 1, x0:x1 + 1, :].copy())

        if use_video_ref:
            gt_fr = gt_frames_rgb[i]
            gt_crop = Image.fromarray(gt_fr[y0:y1 + 1, x0:x1 + 1, :].copy())
            gen_feat = _encode_pil(gen_crop, backend, clip_model, preprocess, device)
            gt_feat  = _encode_pil(gt_crop,  backend, clip_model, preprocess, device)
            score = float((gen_feat @ gt_feat.T).item())
        else:
            gen_feat = _encode_pil(gen_crop, backend, clip_model, preprocess, device)
            score = float((ref_feat @ gen_feat.T).item())

        per_scores.append(score)

        if output_vis_path is not None and has_cv2_writer:
            vis = _draw_bbox_and_score(fr, bbox, score)
            vis_writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    if output_vis_path is not None and has_cv2_writer:
        vis_writer.release()

    # 6) 聚合
    scores_np = np.array(per_scores, dtype=np.float32)
    valid = ~np.isnan(scores_np)
    n_valid = int(valid.sum())
    if n_valid == 0:
        overall = float("nan")
    else:
        overall = float(np.nanmean(scores_np) if aggregate == "mean" else np.nanmedian(scores_np))

    return {
        "overall": overall,
        "per_frame": per_scores,
        "n_frames": n,
        "n_valid": n_valid,
        "vis_path": output_vis_path if output_vis_path is not None else None,
    }
