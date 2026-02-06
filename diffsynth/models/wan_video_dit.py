import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange, repeat
import matplotlib.pyplot as plt
# import seaborn as sns
import os, math
from .utils import hash_state_dict_keys
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end +1, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    # import pdb; pdb.set_trace()
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight

class GateLayer(nn.Module):
    def __init__(self, d, hidden_dim=None,device="cuda",dtype=torch.bfloat16):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d // 2  # 可以自己设，减少参数量
        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim, 1, dtype=dtype)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [b, head, s, d]
        b, h, s, d = x.shape
        # import pdb; pdb.set_trace()
        # 先过 MLP -> [b, head, s, 1]
        out = self.mlp(x)  
        # 在序列维度聚合，比如平均
        out = out.mean(dim=-2, keepdim=True)  # [b, head, 1, 1]
        # 映射到 [0,1]
        gate = self.sigmoid(out)  # [b, head, 1, 1]
        # import pdb; pdb.set_trace()
        return gate

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def viz_attn_bias_from_tensor(
    attn_bias: torch.Tensor,     # [B, 1 or H, Sq, Sk]  加性mask：0=允许, 负大数=禁止
    out_path_base: str,
    vis_batch: int = 0,
    vis_head: int = 0,           # 若第二维是 H，选一个 head；若为1则忽略
    ds_q: int = 64,
    ds_k: int = 64,
    ref_key_start: int | None = None,  # 如果 full_k=[k_main|k_ref] 想只高亮右半，可填右半起点索引
):
    os.makedirs(os.path.dirname(out_path_base) or ".", exist_ok=True)

    # 1) 选择 batch / head，并搬到 CPU
    ab = attn_bias.detach()
    if ab.is_cuda:
        ab = ab.to("cpu")
    B, C, Sq, Sk = ab.shape   # C = 1 or H
    b = max(0, min(vis_batch, B-1))
    h = 0 if C == 1 else max(0, min(vis_head, C-1))
    ab2 = ab[b, h]            # [Sq, Sk]  现在是真二维

    # 2) 阈值化为布尔屏蔽（True=被屏蔽）
    #    注意：有时 dtype 是 bf16/f16，负无穷可能存成大负数，这里统一用 < 0 判断
    masked = (ab2 < 0)

    # 3) 下采样（用等距采样避免巨阵）
    Sq_ds = int(min(Sq, max(1, ds_q)))
    Sk_ds = int(min(Sk, max(1, ds_k)))
    q_idx = torch.linspace(0, max(Sq-1,0), steps=Sq_ds, dtype=torch.long)
    k_idx = torch.linspace(0, max(Sk-1,0), steps=Sk_ds, dtype=torch.long)
    small = masked.index_select(0, q_idx).index_select(1, k_idx)  # [Sq', Sk']

    # 4) 如果想把 full_k 中的右半(ref)强调出来（只是视觉分割用）
    if ref_key_start is not None:
        r0 = int(round(ref_key_start * (Sk_ds / Sk)))  # 把原始列号映射到下采样坐标
        r0 = max(0, min(r0, Sk_ds-1))
        # 画两块背景色带（可选）
        # 这里只是演示：真正画的时候我们直接加竖线/标题说明更清爽

    # 5) 画图
    arr = small.to(torch.float32).numpy()  # 1=masked, 0=allowed
    plt.figure(figsize=(7, 5))
    im = plt.imshow(arr, cmap="gray_r", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    title = "attn_bias mask (1=masked, 0=allowed)"
    if C == 1:
        title += " [shared across heads]"
    else:
        title += f" [head={h}]"
    plt.title(title)
    plt.xlabel("Keys (downsampled)")
    plt.ylabel("Queries (downsampled)")
    if ref_key_start is not None:
        # 画一条虚线分隔 main/ref（注意使用下采样后的列号）
        r0 = int(round(ref_key_start * (Sk_ds / Sk)))
        r0 = max(0, min(r0, Sk_ds-1))
        plt.axvline(x=r0-0.5, linestyle="--", linewidth=1.0, color="white", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{out_path_base}_attnBias_fromTensor.png", dpi=300, bbox_inches="tight")
    plt.close()


def _viz_attn_bias_light(
    out_path_base: str,
    mK: torch.Tensor | None,      # 允许的 key 掩码；支持 [B,Sk] / [B,1,Sk] / [B,1,1,Sk]
    mQ: torch.Tensor | None,      # 允许的 query 掩码；支持 [B,Sq] / [B,1,Sq] / [B,1,Sq,1]
    ref_key_start: int | None,    # 若在 full_k 上只想屏蔽右半(ref)，给出右半起始列；否则传 None
    Sq: int, Sk: int,
    vis_batch: int = 0,
    ds_q: int = 64, ds_k: int = 64,
):
    os.makedirs(os.path.dirname(out_path_base) or ".", exist_ok=True)

    def _to_allowed_1d(mask: torch.Tensor | None, target_len: int, name: str):
        if mask is None:
            return None
        # 统一到 bool，移到 CPU
        m = mask.to(torch.bool).detach().cpu()
        # 压掉 batch 之外的单例维度，最后 reshape 到 [B, target_len]
        # 典型输入：
        #   key: [B,1,1,Sk] / [B,1,Sk] / [B,Sk]
        #   query: [B,1,Sq,1] / [B,1,Sq] / [B,Sq]
        if m.dim() < 2:
            raise ValueError(f"{name} dim<{2}: got {tuple(m.shape)}")
        B = m.shape[0]
        # 连续挤掉除 batch 外的 1 维
        while m.dim() > 2 and (m.shape[1] == 1 or m.shape[-1] == 1):
            # 优先去掉中间/末尾的单例维
            if m.shape[-1] == 1:
                m = m.squeeze(-1)
            elif m.shape[1] == 1:
                m = m.squeeze(1)
        # 现在 m 可能是 [B,L] 或 [B,1,L]，再做一次挤压
        if m.dim() == 3 and m.shape[1] == 1:
            m = m.squeeze(1)
        # 最终 reshape 到 [B, target_len]
        m = m.reshape(B, -1)
        if m.shape[1] != target_len:
            raise ValueError(f"{name} last dim {m.shape[1]} != target {target_len} (raw shape={tuple(mask.shape)})")
        return m  # [B, target_len] ，True=允许

    mK1 = _to_allowed_1d(mK, Sk, "mK")  # [B,Sk] or None
    mQ1 = _to_allowed_1d(mQ, Sq, "mQ")  # [B,Sq] or None

    B = 1 if (mK1 is None and mQ1 is None) else (mK1.shape[0] if mK1 is not None else mQ1.shape[0])
    b = max(0, min(vis_batch, B-1))

    # 下采样索引
    Sq_ds = int(min(Sq, max(1, ds_q)))
    Sk_ds = int(min(Sk, max(1, ds_k)))
    q_idx = torch.linspace(0, max(Sq-1, 0), steps=Sq_ds, dtype=torch.long)
    k_idx = torch.linspace(0, max(Sk-1, 0), steps=Sk_ds, dtype=torch.long)

    # 允许标记（True=允许）
    q_ok = (torch.ones(Sq, dtype=torch.bool)[q_idx] if mQ1 is None else mQ1[b][q_idx])
    k_ok = (torch.ones(Sk, dtype=torch.bool)[k_idx] if mK1 is None else mK1[b][k_idx])

    # 组合逻辑：默认被屏蔽 = (~k_ok) OR (~q_ok)
    mask = (~k_ok)[None, :].clone().expand(q_ok.numel(), -1)  # [Sq',Sk']
    mask |= (~q_ok)[:, None]                                  # [Sq',Sk']

    # 若只想让“右半（ref）”受行屏蔽，左半只受列屏蔽：
    if ref_key_start is not None:
        r0 = int(ref_key_start)
        right = (k_idx >= r0)
        left  = ~right
        if left.any():
            # 左半：仅列屏蔽
            mask[:, left] = (~k_ok[left])[None, :].expand(q_ok.numel(), left.sum().item())

    arr = mask.to(torch.float32).numpy()  # 1=屏蔽, 0=允许

    plt.figure(figsize=(7, 5))
    im = plt.imshow(arr, cmap="gray_r", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("attn_mask (1=masked, 0=allowed)")
    plt.xlabel("Keys (downsampled)")
    plt.ylabel("Queries (downsampled)")
    plt.tight_layout()
    plt.savefig(f"{out_path_base}_attnBias_mask.png", dpi=300, bbox_inches="tight")
    plt.close()



def _viz_attn_bias(
    attn_bias: torch.Tensor,            # [B, 1, Sq, Sk]（或 [B, H, Sq, Sk] 也兼容）
    out_path_base: str,                 # 基础文件名（不带后缀）
    vis_batch: int = 0,
    ds_q: int = 32,                     # Query 维下采样步长
    ds_k: int = 32,                     # Key   维下采样步长
):
    """
    将 additive attn_bias 直接可视化：
      - 生成 *_attnBias_mask.png：把 <0 的位置视为“被屏蔽”，显示 1；其余为 0。
      - 生成 *_attnBias_values.png：显示原始 bias 值（高动态范围，可看 0 和 -1e9 分布）。
      - 生成 *_attnBias_rowMask.png / *_attnBias_colMask.png：行/列是否被整体屏蔽（方便确认 query 或 key 的整行/整列禁用）。
    """
    os.makedirs(os.path.dirname(out_path_base) or ".", exist_ok=True)

    # 兼容 [B,1,Sq,Sk] 或 [B,H,Sq,Sk]；这里我们取第一个 head（若存在）
    if attn_bias.dim() != 4:
        raise ValueError(f"attn_bias must be 4D, got shape={tuple(attn_bias.shape)}")
    B, H1, Sq, Sk = attn_bias.shape
    b = max(0, min(vis_batch, B - 1))

    # 取 [Sq, Sk]
    # 如果 H1>1（已广播到 head 维），随便取一个 head（0）
    bias_2d = attn_bias[b, 0]                      # [Sq, Sk]
    bias_2d_cpu = bias_2d.detach().to(torch.float32).cpu()

    # 下采样索引
    q_idx = torch.arange(0, Sq, max(1, int(ds_q)), dtype=torch.long)
    k_idx = torch.arange(0, Sk, max(1, int(ds_k)), dtype=torch.long)

    small = bias_2d_cpu.index_select(0, q_idx).index_select(1, k_idx)   # [Sq', Sk']

    # --- (A) 屏蔽掩码图（<0 视为屏蔽）
    mask = (small < 0).to(torch.float32).numpy()                         # 0/1
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap="gray_r", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("attn_bias mask (1=blocked, 0=allowed)")
    plt.xlabel("K (downsampled)")
    plt.ylabel("Q (downsampled)")
    plt.tight_layout()
    plt.savefig(f"{out_path_base}_attnBias_mask.png", dpi=400, bbox_inches="tight")
    plt.close()

    # --- (B) 原始数值热图（0 / -1e9 等）
    arr = small.numpy()
    # 为了色彩可读性，裁剪到[-1.1e9, 0]
    vmin, vmax = -1.1e9, 0.0
    plt.figure(figsize=(8, 6))
    plt.imshow(np.clip(arr, vmin, vmax), cmap="magma", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("attn_bias values (clipped)")
    plt.xlabel("K (downsampled)")
    plt.ylabel("Q (downsampled)")
    plt.tight_layout()
    plt.savefig(f"{out_path_base}_attnBias_values.png", dpi=400, bbox_inches="tight")
    plt.close()

    # --- (C) 行/列整体屏蔽条带（全列/全行都 <0）
    full_mask = (bias_2d_cpu < 0).to(torch.bool).numpy()  # [Sq, Sk] (未下采样，用全量更准确)
    row_blocked = full_mask.all(axis=1).astype(np.float32)[None, :]  # [1, Sq]
    col_blocked = full_mask.all(axis=0).astype(np.float32)[None, :]  # [1, Sk]

    # 行条带
    plt.figure(figsize=(10, 1.6))
    plt.imshow(row_blocked, cmap="gray_r", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    plt.yticks([]); plt.xlabel("Query index"); plt.title("Row mask (1=entire row blocked)")
    plt.tight_layout()
    plt.savefig(f"{out_path_base}_attnBias_rowMask.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 列条带
    plt.figure(figsize=(10, 1.6))
    plt.imshow(col_blocked, cmap="gray_r", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    plt.yticks([]); plt.xlabel("Key index"); plt.title("Column mask (1=entire column blocked)")
    plt.tight_layout()
    plt.savefig(f"{out_path_base}_attnBias_colMask.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 也保存一份原始 bias 的下采样 numpy，便于排查
    np.save(f"{out_path_base}_attnBias_small.npy", arr)


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    vmin, vmax = float(np.min(x)), float(np.max(x))
    if vmax <= vmin + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - vmin) / (vmax - vmin)

def _save_heatmap2d(arr2d: np.ndarray, out_path: str, title: str = "", dpi=200, cmap="magma"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    arr2d = _minmax01(arr2d)
    plt.figure(figsize=(6, 5))
    plt.imshow(arr2d, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.colorbar(fraction=0.046, pad=0.04)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()

def _tile_frames(frames: list[np.ndarray], ncols: int = 8, pad: int = 2) -> np.ndarray:
    """把一组 HxW 的灰度帧拼成马赛克。"""
    if len(frames) == 0:
        return None
    h, w = frames[0].shape
    ncols = max(1, ncols)
    nrows = (len(frames) + ncols - 1) // ncols
    canvas = np.ones((nrows * h + (nrows - 1) * pad,
                      ncols * w + (ncols - 1) * pad), dtype=np.float32)
    canvas *= 0.0
    for idx, fr in enumerate(frames):
        r, c = divmod(idx, ncols)
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0:y0+h, x0:x0+w] = fr
    return canvas


import numpy as np
import torch, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _save_volume_as_grid(volume_3d, out_path, dpi=400, cmap="magma"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    F, H, W = volume_3d.shape
    vmin = np.percentile(volume_3d, 5.0)
    vmax = np.percentile(volume_3d, 99.5)
    vmax = max(vmax, vmin + 1e-6)

    cols = int(math.ceil(math.sqrt(F)))
    rows = int(math.ceil(F / cols))
    fig_w = max(12, cols * 2.5)
    fig_h = max(8,  rows * 2.5)

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(rows, cols)

    idx = 0
    im = None
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            ax.axis("off")
            if idx < F:
                im = ax.imshow(volume_3d[idx], cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(f"t={idx}")
                idx += 1

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def _visualize_query_sampleK_3d_all_heads_cpu(
    qh: torch.Tensor, kh: torch.Tensor,              # [B,H,Sq,D], [B,H,Sk,D]
    out_path_base: str, grid_size: tuple,            # (f,h,w) —— 对应 Query 侧网格
    vis_batch: int = 0, vis_heads: list | None = None,
    sample_key_index: int | None = None,
    sample_per_head: bool = False,
    use_softmax: bool = True,
    q_chunk: int = 1024,
    key_downsample: int = 1,
    seed: int | None = None,
    key_token_mask: torch.Tensor | None = None,      # 列遮罩：允许作为 Key 的位置（1=可用、0=禁用）
    query_token_mask: torch.Tensor | None = None,    # 行遮罩：允许从 ref 取值的 Query（1=可用、0=禁用）
    **_,                                             # 兜底：避免老代码传入未知关键字时报错
):
    """
    可视化思路：
      - 先随机（或指定）选择一个 Key 列 kk；
      - 计算所有 Query 对该列的注意力（logits->softmax 后该列的概率），得到 qAgg ∈ [Sq]；
      - 将 qAgg 还原成 (f,h,w) 并保存 3D 马赛克图；
      - 若传入 query_token_mask，则把被禁用的 Query 行直接置 0，让 3D 图明显出现“黑块/黑条”；
      - 若传入 key_token_mask，则采样列时只从允许的 Key 中选，并在 softmax 前把禁用列设为 -1e9。
    """
    os.makedirs(os.path.dirname(out_path_base) or ".", exist_ok=True)

    # ---- 形状与网格检查 ----
    B, H, Sq, D = qh.shape
    Sk = kh.shape[2]
    f, h, w = [int(x) for x in grid_size]
    assert f * h * w == Sq, f"grid_size 与 Sq 不一致：f*h*w={f*h*w}, Sq={Sq}"

    # ---- 选 batch、头 ----
    b = max(0, min(vis_batch, B - 1))
    heads = list(range(H)) if vis_heads is None else [hh for hh in vis_heads if 0 <= hh < H]

    # ---- 搬到 CPU，float32 便于数值操作 ----
    q_b = qh[b].detach().to(torch.float32).cpu()     # [H,Sq,D]
    k_b = kh[b].detach().to(torch.float32).cpu()     # [H,Sk,D]

    # ---- 处理 Key 列遮罩（支持 [B,Sk] / [B,1,Sk] / [B,H,Sk]）----
    per_head_key_mask = False
    m_key = None
    if key_token_mask is not None:
        m = key_token_mask
        if m.dim() == 2:             # [B,Sk]
            m_key = m[b].unsqueeze(0)    # -> [1,Sk]
        elif m.dim() == 3 and m.size(1) == 1:  # [B,1,Sk]
            m_key = m[b]                     # -> [1,Sk]
        elif m.dim() == 3 and m.size(1) == H:  # [B,H,Sk]
            m_key = m[b]                     # -> [H,Sk]
            per_head_key_mask = True
        else:
            raise ValueError(f"key_token_mask 形状非法：{tuple(m.shape)}，应为 [B,Sk] / [B,1,Sk] / [B,H,Sk]")
        m_key = m_key.to(torch.bool).cpu()   # bool

    # ---- 处理 Query 行遮罩（支持 [B,Sq] / [B,1,Sq] / [B,H,Sq]）----
    per_head_query_mask = False
    m_query = None
    if query_token_mask is not None:
        mq = query_token_mask
        if mq.dim() == 2:               # [B,Sq]
            m_query = mq[b].unsqueeze(0)   # -> [1,Sq]
        elif mq.dim() == 3 and mq.size(1) == 1:  # [B,1,Sq]
            m_query = mq[b]                    # -> [1,Sq]
        elif mq.dim() == 3 and mq.size(1) == H: # [B,H,Sq]
            m_query = mq[b]                    # -> [H,Sq]
            per_head_query_mask = True
        else:
            raise ValueError(f"query_token_mask 形状非法：{tuple(mq.shape)}，应为 [B,Sq] / [B,1,Sq] / [B,H,Sq]")
        assert m_query.shape[-1] == Sq, f"query_token_mask 的 Sq 维不匹配：{m_query.shape[-1]} != {Sq}"
        m_query = m_query.to(torch.bool).cpu()     # bool

    # ---- Key 下采样（保持遮罩同步下采样）----
    if key_downsample > 1:
        idx_k = torch.arange(0, Sk, key_downsample, dtype=torch.long)
        k_b   = k_b.index_select(1, idx_k)       # [H,Sk',D]
        Sk    = k_b.shape[1]
        if m_key is not None:
            m_key = m_key.index_select(1, idx_k) # [1或H, Sk']

    # ---- 随机数发生器（可复现）----
    if seed is not None:
        g = torch.Generator(); g.manual_seed(int(seed))
        rand_int = lambda hi: int(torch.randint(0, hi, (1,), generator=g).item())
    else:
        rand_int = lambda hi: int(torch.randint(0, hi, (1,)).item())

    scale = math.sqrt(D)

    for hh in heads:
        # === 取当前 head 的列、行遮罩 ===
        key_mask_vec   = None if m_key   is None else (m_key[hh] if per_head_key_mask   else m_key[0])   # [Sk] or None
        query_mask_vec = None if m_query is None else (m_query[hh] if per_head_query_mask else m_query[0]) # [Sq] or None

        # === 从允许的 key 中采样 kk（或使用指定列）===
        if key_mask_vec is not None:
            allowed = torch.nonzero(key_mask_vec, as_tuple=False).squeeze(-1)
            if allowed.numel() == 0:
                # 该 head 没有任何可用列，跳过
                continue
        else:
            allowed = None

        if sample_key_index is None:
            if sample_per_head:
                kk = (allowed[rand_int(allowed.numel())].item()
                      if allowed is not None else rand_int(Sk))
            else:
                if hh == heads[0]:
                    kk = (allowed[rand_int(allowed.numel())].item()
                          if allowed is not None else rand_int(Sk))
        else:
            kk = int(sample_key_index)
            if allowed is not None and not key_mask_vec[kk].item():
                # 若用户给的 kk 不在允许集合里，就用 allowed[0]
                kk = allowed[0].item()

        # === 分块计算 scores，并施加列遮罩；随后取第 kk 列 ===
        qAgg = torch.empty(Sq, dtype=torch.float32)
        k_h  = k_b[hh]                   # [Sk,D]
        k_t  = k_h.t()                   # [D,Sk]

        s = 0
        while s < Sq:
            e = min(s + q_chunk, Sq)
            q_slice = q_b[hh, s:e, :]           # [cq,D]
            scores  = (q_slice @ k_t) / scale   # [cq,Sk]

            if use_softmax:
                if key_mask_vec is not None:
                    scores[:, ~key_mask_vec] = -1e9  # 列屏蔽
                probs = torch.softmax(scores, dim=-1)
                vec = probs[:, kk]                   # [cq]
            else:
                vec = scores[:, kk]

            # === 行遮罩可视化：把被禁用的行清零，3D 图上就会是黑块 ===
            if query_mask_vec is not None:
                q_slice_mask = query_mask_vec[s:e]   # [cq] bool
                vec = vec * q_slice_mask.to(vec.dtype)

            qAgg[s:e] = vec
            s = e

        # 再保守一遍（以防上面没有分块覆盖到某些逻辑）
        if query_mask_vec is not None and query_mask_vec.numel() == qAgg.numel():
            qAgg[~query_mask_vec] = 0.0

        # === 还原 3D 并保存 ===
        vol = qAgg.view(f, h, w).numpy()
        out_img = f"{out_path_base}_H{hh:02d}_k{kk}_qAgg3d.png"
        _save_volume_as_grid(vol, out_img, dpi=400, cmap="magma")
        np.save(f"{out_path_base}_H{hh:02d}_k{kk}_qAgg3d.npy", vol)

        # === 附加：把行遮罩也存成 3D 马赛克，便于目视对比（可选）===
        if query_mask_vec is not None:
            qmask_vol = query_mask_vec.view(f, h, w).to(torch.float32).numpy()
            _save_volume_as_grid(qmask_vol, f"{out_path_base}_H{hh:02d}_qmask3d.png", dpi=300, cmap="gray_r")

        # === 附加：把 key 列遮罩存成 1D 条带图（可选）===
        if key_mask_vec is not None:
            bar = key_mask_vec.to(torch.float32).numpy()[None, :]  # [1,Sk]
            plt.figure(figsize=(10, 1.6))
            plt.imshow(bar, cmap="gray_r", aspect="auto", vmin=0, vmax=1, interpolation="nearest")
            plt.yticks([]); plt.xlabel("Key index"); plt.title("Key mask (1=allowed, 0=blocked)")
            plt.tight_layout()
            plt.savefig(f"{out_path_base}_H{hh:02d}_keymask1d.png", dpi=300, bbox_inches="tight")
            plt.close()

import scipy.ndimage

import torch
import matplotlib.pyplot as plt
import scipy.ndimage
from einops import rearrange

import torch
import matplotlib.pyplot as plt
import scipy.ndimage

def visualize_attention_weights(attn_weights: torch.Tensor, vis_out: str, vis_batch: int = 0, vis_head: int = 0, downsample_factor: int = 4, clip_value: float = 1e3):
    """
    Visualize the attention weights after softmax, with optional downsampling and clipping for extreme values.
    """
    # Ensure the tensor is on CPU and detach from the computation graph
    attn_weights = attn_weights[vis_batch, vis_head].cpu().detach().float()

    # Debug: Print the shape of the tensor
    print(f"Shape of attention weights for batch {vis_batch}, head {vis_head}: {attn_weights.shape}")

    # Clip extreme values to improve visualization (e.g., values beyond ±1e3)
    # attn_weights = torch.clamp(attn_weights, min=-clip_value, max=clip_value)

    # Apply softmax to the attention weights to normalize them into [0, 1] range
    attn_weights = torch.softmax(attn_weights, dim=-1)

    # Debug: Print the shape of the tensor after softmax
    print(f"Shape of attention weights after softmax: {attn_weights.shape}")

    # # Downsample (缩放) attention map for faster visualization
    # if downsample_factor > 1:
    #     # Make sure to check if the downsampling is applicable to the correct dimensions
    #     attn_weights = scipy.ndimage.zoom(attn_weights.numpy(), (downsample_factor, downsample_factor), order=1)

    # Plot the attention weights as a heatmap
    print(f"{vis_out}_batch{vis_batch}_head{vis_head}_weights.png")
    plt.imshow(attn_weights, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Attention Weights (Batch {vis_batch}, Head {vis_head})")
    plt.savefig(f"{vis_out}_batch{vis_batch}_head{vis_head}_weights.png", bbox_inches='tight')
    plt.close()

# Example usage:
# visualize_attention_weights(attn_weights, 'path_to_output', vis_batch=0, vis_head=0)






def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    num_heads: int, compatibility_mode: bool = False,
                    visualize: bool = False,
                    vis_out: str = "attention",
                    vis_batch: int = 0, vis_head: int = 0,
                    vis_downsample: int = 1,
                    query_grid_size: tuple | None = None,
                    sample_k_column: bool = True,
                    sample_key_index: int | None = None,
                    sample_per_head: bool = False,
                    use_softmax_for_sample: bool = True,
                    q_chunk: int = 1024,
                    key_downsample: int = 1,
                    sample_seed: int | None = None,
                    # ===== NEW: 支持 key 方向的 attn mask =====
                    key_token_mask: torch.Tensor | None = None,
                    query_token_mask: torch.Tensor | None = None,
                    ref_key_start: int | None = None,
                    key_grid_size: tuple | None = None,
                    heat_ds_q: int = 32, heat_ds_k: int = 32):
    """
    若提供 key_token_mask（形状 [B, Sk] 或 [B, 1, Sk]，1=允许注意，0=禁止），
    则强制使用 PyTorch SDPA，并通过 additive attn_mask 限制注意力到指定 keys。
    """
    # 只要任意一种 mask 存在，就强制走 SDPA
    use_mask = (key_token_mask is not None) or (query_token_mask is not None)

    def _to_BHSD(t):
        return rearrange(t, "b s (n d) -> b n s d", n=num_heads)

    if use_mask:
        qh, kh, vh = _to_BHSD(q), _to_BHSD(k), _to_BHSD(v)  # [B,H,Sq,D]
        B, H, Sq, D = qh.shape
        Sk = kh.shape[2]

        # 统一 & 校验
        # --- Key mask: [B,Sk] 或 [B,1,Sk]
        if key_token_mask is not None:
            mK = key_token_mask
            if mK.dim() == 2: mK = mK.unsqueeze(1)   # [B,1,Sk]
            assert mK.shape[0] == B and mK.shape[-1] == Sk
            disallow_k = (~mK.to(torch.bool)).to(qh.dtype).unsqueeze(2)  # [B,1,1,Sk]
        else:
            disallow_k = None

        # --- Query mask: [B,Sq] 或 [B,1,Sq]
        if query_token_mask is not None:
            mQ = query_token_mask
            if mQ.dim() == 2: mQ = mQ.unsqueeze(1)   # [B,1,Sq]
            assert mQ.shape[0] == B and mQ.shape[-1] == Sq
            disallow_q = (~mQ.to(torch.bool)).to(qh.dtype).unsqueeze(-1) # [B,1,Sq,1]
        else:
            disallow_q = None

        # 合成 additive bias
        attn_bias = torch.zeros(B, 1, Sq, Sk, dtype=qh.dtype, device=qh.device)
        if disallow_k is not None: attn_bias = attn_bias + disallow_k * (-1e9)
        # if disallow_q is not None: attn_bias = attn_bias + disallow_q * (-1e9)
        if disallow_q is not None:
            if ref_key_start is None:
                # 旧行为：整行禁用（对所有列）
                attn_bias = attn_bias + disallow_q * (-1e9)
            else:
                # 新行为：只对 [ref_key_start : Sk) 这段列禁用
                r0 = int(ref_key_start)
                if r0 < Sk:
                    Sref = Sk - r0
                    attn_bias[..., r0:] = attn_bias[..., r0:] + \
                        disallow_q.expand(B, 1, Sq, Sref) * (-1e9)


        # 前向
        xh = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=attn_bias)  # [B,H,Sq,D]
        # xh = F.scaled_dot_product_attention(qh, kh, vh)  # [B,H,Sq,D]


        # === 调试：可视化 additive attn_bias ===
        # Visualization of attention map
        if visualize:
            # def _to_BHSD(t):
            #     return rearrange(t, "b s (n d) -> b n s d", n=num_heads)

            # qh, kh, vh = _to_BHSD(q), _to_BHSD(k), _to_BHSD(v)  # [B,H,Sq,D]
            # B, H, Sq, D = qh.shape
            # Sk = kh.shape[2]

            # 计算 attention 权重
            qh = qh.view(B, H, Sq, D) # [B, H, Sq, D]
            kh = kh.view(B, H, Sk, D)  # [B, H, Sk, D]
            attn_weights = torch.matmul(qh, kh.transpose(-2, -1)) / torch.sqrt(torch.tensor(D, dtype=torch.float32))  # [B, H, Sq, Sk]
            
            # 如果有 mask，我们会应用
            if attn_bias is not None:
                attn_weights = attn_weights.masked_fill(attn_bias<-1e5, float("-inf"))

            if visualize:
                try:
                    for hh in range(H):
                        visualize_attention_weights(
                            attn_weights,
                            vis_out=vis_out,
                            vis_batch=vis_batch,
                            vis_head=hh,
                            downsample_factor=heat_ds_q
                        )
                except Exception as e:
                    print(f"[visualize-attn-weights] skipped: {type(e).__name__}: {e}")

                try:
                    base = os.path.splitext(vis_out)[0]
                    # 注意：为了省显存，用 CPU 作图
                    viz_attn_bias_from_tensor(
                        attn_bias.to("cpu"),  # 避免在 GPU 物化大矩阵
                        out_path_base=base,
                        vis_batch=vis_batch,
                        vis_head=0,
                        ds_q=10000, ds_k=10000,
                        ref_key_start=None,   # 如果 full_k=[k_main|k_ref] 并想分隔可填右半起始列
                    )
                except Exception as e:
                    print(f"[viz-attn-bias] skipped: {type(e).__name__}: {e}")

        x = rearrange(xh, "b n s d -> b s (n d)", n=num_heads)
    else:
        # print("FLASH ATTN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # ====== 原有分支（无 mask）保持不变 ======
        if compatibility_mode:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            qh, kh = q, k
            x = F.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)

        elif FLASH_ATTN_3_AVAILABLE:
            q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
            qh, kh = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)
            x = flash_attn_interface.flash_attn_func(q, k, v)
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)

        elif FLASH_ATTN_2_AVAILABLE:
            q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
            qh, kh = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)
            x = flash_attn.flash_attn_func(q, k, v)
            x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)

        elif SAGE_ATTN_AVAILABLE:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            qh, kh = q, k
            x = sageattn(q, k, v)
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)

        else:
            q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
            k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
            v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
            qh, kh = q, k
            x = F.scaled_dot_product_attention(q, k, v)
            x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)

    # # == 下面保留你之前的可视化（省略） ==
    # # if visualize: ...（不变）
    # if visualize and query_grid_size is not None and sample_k_column:
    #     try:
    #         base = os.path.splitext(vis_out)[0]
    #         _visualize_query_sampleK_3d_all_heads_cpu(
    #             qh, kh,
    #             out_path_base=base,
    #             grid_size=query_grid_size,
    #             vis_batch=vis_batch,
    #             vis_heads=None,
    #             sample_key_index=sample_key_index,
    #             sample_per_head=sample_per_head,
    #             use_softmax=use_softmax_for_sample,
    #             q_chunk=q_chunk,
    #             key_downsample=key_downsample,
    #             seed=sample_seed,
    #             key_token_mask=key_token_mask, 
    #             query_token_mask=query_token_mask
    #         )
    #     except Exception as ve:
    #         print(f"[flash_attention][sampleK-3D] skipped: {type(ve).__name__}: {ve}")

    return x




import os, math, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).cpu().numpy()
    a = np.asarray(x)
    return a.astype(np.float32, copy=False)


class AttentionModule(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim: int,
        layer_id: int = 0,
        vis_dir: str = "./tmp/attn",
        visualize: bool = False,      # 默认是否可视化
        vis_head: int = 0,            # 仍保留（2D用到时）
        vis_batch: int = 0,
        vis_downsample: int = 1,
        # ===== 3D 可视化（K维采样一列）的控制项 =====
        sample_k_column: bool = True,         # 开启“按K采样一列→(f,h,w)”可视化
        sample_key_index: Optional[int] = None,  # 固定列；None=随机
        sample_per_head: bool = False,        # True=每个head各自随机一列
        use_softmax_for_sample: bool = True,  # True=按概率列；False=logits列
        q_chunk: int = 1024000000000,                  # Query 分块（防 OOM）
        key_downsample: int = 1,              # Key 下采样（>1 近似）
        sample_seed: Optional[int] = None,    # 固定随机种子
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.layer_id = layer_id
        self.vis_dir = vis_dir
        self.visualize = visualize
        self.vis_head = vis_head
        self.vis_batch = vis_batch
        self.vis_downsample = vis_downsample

        # 3D 相关
        self.grid_size: Optional[Tuple[int, int, int]] = None  # 外部设置 (f,h,w)
        self.sample_k_column = sample_k_column
        self.sample_key_index = sample_key_index
        self.sample_per_head = sample_per_head
        self.use_softmax_for_sample = use_softmax_for_sample
        self.q_chunk = q_chunk
        self.key_downsample = key_downsample
        self.sample_seed = sample_seed

        self._call_idx = 0
        if self.visualize:
            os.makedirs(self.vis_dir, exist_ok=True)

    def init_gate(self,
                  train: bool = False,
                  dtype=torch.bfloat16,
                  mode: str = "neutral",
                  p: float = 0.99,
                  noise_std: float = 0.0,
                  tau: float = 1.0):
        self.tau = tau
        self.gate = GateLayer(d=self.dim // self.num_heads)

        assert mode in ("neutral", "open", "closed"), "mode must be 'neutral'|'open'|'closed'"
        requires_grad = train

        modules = list(self.gate.mlp)
        first_linear = None
        last_linear = None
        for m in modules:
            if isinstance(m, nn.Linear):
                if first_linear is None:
                    first_linear = m
                last_linear = m

        if first_linear is not None:
            nn.init.xavier_uniform_(first_linear.weight)
            if first_linear.bias is not None:
                nn.init.zeros_(first_linear.bias)

        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                if mode == "neutral":
                    bias_val = 0.0
                else:
                    p_used = min(max(float(p), 1e-6), 1.0 - 1e-6)
                    logit_p = math.log(p_used / (1.0 - p_used))
                    bias_val = logit_p * float(self.tau)

                if noise_std > 0.0:
                    with torch.no_grad():
                        b = torch.full_like(last_linear.bias.data, float(bias_val))
                        b += torch.randn_like(b) * float(noise_std)
                        last_linear.bias.data.copy_(b)
                else:
                    nn.init.constant_(last_linear.bias, float(bias_val))

        for m in modules:
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 最后统一转 dtype
        try:
            self.gate.to(dtype)
        except Exception:
            # 若环境不支持 bfloat16，则退回到 float32（不会抛崩溃）
            self.gate.to(torch.float32)

        # 设置 requires_grad
        for p_param in self.gate.parameters():
            p_param.requires_grad = requires_grad


    def forward(self, q, k, v, visualize: bool = False, key_token_mask: torch.Tensor | None = None, query_token_mask: torch.Tensor | None = None, ref_key_start: int | None = None):
        orig_dtype = q.dtype
        do_vis = visualize
        if do_vis:
            self._call_idx += 1
            os.makedirs(self.vis_dir, exist_ok=True)
            out_base = os.path.join(self.vis_dir, f"layer{self.layer_id}_call{self._call_idx:06d}")
        else:
            out_base = ""

        x = flash_attention(
            q=q.to(torch.bfloat16),
            k=k.to(torch.bfloat16),
            v=v.to(torch.bfloat16),
            num_heads=self.num_heads,
            visualize=do_vis,
            vis_out=out_base,
            vis_batch=self.vis_batch,
            vis_head=self.vis_head,
            vis_downsample=self.vis_downsample,
            query_grid_size=self.grid_size,
            sample_k_column=self.sample_k_column,
            sample_key_index=self.sample_key_index,
            sample_per_head=self.sample_per_head,
            use_softmax_for_sample=self.use_softmax_for_sample,
            q_chunk=self.q_chunk,
            key_downsample=self.key_downsample,
            sample_seed=self.sample_seed,
            # ==== NEW ====
            key_token_mask=key_token_mask,
            query_token_mask=query_token_mask,   # 新增
            ref_key_start=ref_key_start
        )

        x = rearrange(x, "b s (n d) -> b n s d", n=self.num_heads)
        # import pdb; pdb.set_trace()
        x = self.gate(x) * x
        x = rearrange(x, "b n s d -> b s (n d)", n=self.num_heads)
        return x.to(orig_dtype)



class LoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 128,
        device="cuda",
        # dtype: Optional[torch.dtype] = torch.float32,
        dtype: Optional[torch.dtype] = torch.bfloat16,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)
        return up_hidden_states.to(orig_dtype)

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, layer_idx = 0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        # self.attn = AttentionModule(self.num_heads)
        self.attn = AttentionModule(
            num_heads=self.num_heads,
            dim= dim,
            layer_id=layer_idx,   # 没有就用默认 0
            vis_dir="./tmp/attn",
            visualize=False,              # 开启
            vis_head=0,
            vis_batch=0,
            vis_downsample=1,            # 长序列可调大点，例如 4/8
        )

        self.key_token_mask_ref: Optional[torch.Tensor] = None
        self.kv_cache = None
        self.cond_size = None

    def init_lora(self, train=False):
        dim = self.dim
        self.q_loras = LoRALinearLayer(dim, dim, rank=128)
        self.k_loras = LoRALinearLayer(dim, dim, rank=128)
        self.v_loras = LoRALinearLayer(dim, dim, rank=128)

        requires_grad = train
        for lora in [self.q_loras, self.k_loras, self.v_loras]:
            for param in lora.parameters():
                param.requires_grad = requires_grad
                # import pdb; pdb.set_trace()
                if param.requires_grad:
                    # Upcast LoRA parameters into fp32
                    param.data = param.to(torch.bfloat16)

    def forward(self, x, freqs):
        if self.cond_size is not None:
            if self.kv_cache is None:
                x_main, x_ref = x[:, : -self.cond_size], x[:, -self.cond_size :]
                freqs_main, freqs_ref = freqs

                q_main = self.norm_q(self.q(x_main))
                k_main = self.norm_k(self.k(x_main))
                v_main = self.v(x_main)

                q_main = rope_apply(q_main, freqs_main, self.num_heads)
                k_main = rope_apply(k_main, freqs_main, self.num_heads)

                q_ref = self.norm_q(self.q(x_ref) + self.q_loras(x_ref))
                k_ref = self.norm_k(self.k(x_ref) + self.k_loras(x_ref))
                v_ref = self.v(x_ref) + self.v_loras(x_ref)

                q_ref = rope_apply(q_ref, freqs_ref, self.num_heads)
                k_ref = rope_apply(k_ref, freqs_ref, self.num_heads)
                self.kv_cache = {"k_ref": k_ref.detach(), "v_ref": v_ref.detach()}

                # === 构造 full_k/full_v ===
                full_k = torch.concat([k_main, k_ref], dim=1)
                full_v = torch.concat([v_main, v_ref], dim=1)
                S_main = k_main.shape[1] 

                cond_out = self.attn(q_ref, k_ref, v_ref, visualize=False)  # 与原逻辑一致

                # import pdb; pdb.set_trace()
                ### mod
                main_out = self.attn(
                    q_main, full_k, full_v,
                    visualize=False,
                    # query_token_mask=self.query_mask_ref_only,  # [B, S_main]，1=该 query 允许看 ref
                    # ref_key_start=S_main                        # 只在 ref 段列生效屏蔽
                )

                # import pdb; pdb.set_trace()
                # ref_out = self.attn(
                #     q_main, k_ref, v_ref,
                #     visualize=True,
                #     query_token_mask=self.query_mask_ref_only 
                # )

                # 你原来把 ref_out 拼 cond_out 返回；若想保持原输出也可改回 main_out
                out = torch.concat([main_out, cond_out], dim=1)
                return self.o(out)

            else:
                # cache 命中：只有 main 序列作为 query
                k_ref = self.kv_cache["k_ref"]
                v_ref = self.kv_cache["v_ref"]

                q_main = self.norm_q(self.q(x))
                k_main = self.norm_k(self.k(x))
                v_main = self.v(x)

                q_main = rope_apply(q_main, freqs, self.num_heads)
                k_main = rope_apply(k_main, freqs, self.num_heads)

                # full_k/full_v
                full_k = torch.concat([k_main, k_ref], dim=1)
                full_v = torch.concat([v_main, v_ref], dim=1)
                S_main = k_main.shape[1]

                # import pdb; pdb.set_trace()
                ### mod
                x_main = self.attn(
                    q_main, full_k, full_v,
                    visualize=False,
                    # query_token_mask=self.query_mask_ref_only,
                    # ref_key_start=S_main
                )
                return self.o(x_main)


                # import pdb; pdb.set_trace()
                # x_ref_only = self.attn(
                #     q_main, k_ref, v_ref,
                #     visualize=True,
                #     query_token_mask=self.query_mask_ref_only 
                # )
                # # 返回哪个看你当前调试目标；这里示例返回 full_k 的结果
                # return self.o(x_ref_only)

        else:
            # 无 cond 模式，原逻辑保持（如也想可视化，可设 visualize=True）
            q = self.norm_q(self.q(x))
            k = self.norm_k(self.k(x))
            v = self.v(x)
            q = rope_apply(q, freqs, self.num_heads)
            k = rope_apply(k, freqs, self.num_heads)
            x = self.attn(q, k, v, visualize=False)
            return self.o(x)



class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads, dim)

        if self.has_image_input:
            self.attn_img = AttentionModule(self.num_heads, dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = self.attn_img(q, k_img, v_img)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6, layer_idx=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps, layer_idx)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

                # === 可视化配置 ===
        self.vis_unpatch = True                 # 开关：是否保存 attn_out 的 3D 可视化
        self.vis_dir = "./tmp/attn_tokens"      # 保存目录
        self.layer_id = layer_idx  # 若外部未覆盖，则为 0
        self._vis_counter = 0                   # 次数计数，防止保存过多
        self.vis_max_saves = 5                  # 每层最多保存几次
        self.vis_save_frames = 80                # 每次保存前多少帧（f 维）
        self.vis_mosaic_cols = 4                # 马赛克每行列数


    def forward(self, x, context, t_mod, freqs, x_ref=None, t_mod_ref=None):
        # 在 DiTBlock.forward 里：把每层需要的 mask 都塞进 self.self_attn
        self.self_attn.key_token_mask_ref  = getattr(self, "key_token_mask_ref", None)   # [B, S_ref]   1=允许
        self.self_attn.query_mask_ref_only = getattr(self, "query_mask_ref_only", None)   # [B, S_main]  1=允许

        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        if x_ref is not None:
            (
                shift_msa_ref,
                scale_msa_ref,
                gate_msa_ref,
                shift_mlp_ref,
                scale_mlp_ref,
                gate_mlp_ref,
            ) = (
                self.modulation.to(dtype=t_mod_ref.dtype, device=t_mod_ref.device)
                + t_mod_ref
            ).chunk(6, dim=1)
            input_x_ref = modulate(
                self.norm1(x_ref), shift_msa_ref, scale_msa_ref
            )  # [1, 1024, 5120]
            self.self_attn.cond_size = input_x_ref.shape[1]
            input_x = torch.concat([input_x, input_x_ref], dim=1)
            self.self_attn.kv_cache = None
        
        attn_out = self.self_attn(input_x, freqs)
        if x_ref is not None:
            attn_out, attn_out_ref = (
                attn_out[:, : -self.self_attn.cond_size],
                attn_out[:, -self.self_attn.cond_size :],
            )

        #         # === 在 gate 之前：把 attn_out 按 (f,h,w) 还原并可视化 ===
        # # import pdb; pdb.set_trace()
        # try:
        #     if getattr(self, "vis_unpatch", False) and self._vis_counter < getattr(self, "vis_max_saves", 5):
        #         grid = getattr(self.self_attn.attn, "grid_main", None)  # (f,h,w)，来自外部设置
        #         if grid is not None and isinstance(grid, (tuple, list)) and len(grid) == 3:
        #             f, h_, w_ = int(grid[0]), int(grid[1]), int(grid[2])
        #             N = f * h_ * w_
        #             B, S, C = attn_out.shape  # [B, N, D]
        #             # 安全检查：若长度不一致，尝试裁剪
        #             if S < N:
        #                 print(f"[DiTBlock attn vis] skip: N={N} > S={S}")
        #             else:
        #                 # 仅取 batch=0 以节省 IO；也可循环多 batch
        #                 tok = attn_out[0, :N, :].detach().to(torch.float32).cpu().numpy()  # [N, D]
        #                 vol = tok.reshape(f, h_, w_, C)                                     # [f,h,w,C]
        #                 # 聚合成灰度：用通道 L2 能量（也可改为 mean）
        #                 vol_energy = np.sqrt(np.maximum(np.sum(vol**2, axis=-1), 0.0))      # [f,h,w]
        #                 vol_energy = _minmax01(vol_energy)

        #                 # 逐帧保存（只保存前 vis_save_frames 帧）
        #                 save_frames = int(min(self.vis_save_frames, f))
        #                 out_base = os.path.join(self.vis_dir, f"layer{self.layer_id:02d}")
        #                 os.makedirs(out_base, exist_ok=True)
        #                 frames_for_mosaic = []
        #                 for t in range(save_frames):
        #                     fr = vol_energy[t]  # [h,w]
        #                     frames_for_mosaic.append(fr)
        #                     out_png = os.path.join(out_base, f"call{self._vis_counter:04d}_f{t:03d}.png")
        #                     _save_heatmap2d(fr, out_png,
        #                                     title=f"layer{self.layer_id} call{self._vis_counter} frame{t}",
        #                                     dpi=200, cmap="magma")
        #                 # 也保存一个马赛克图
        #                 mosaic = _tile_frames(frames_for_mosaic, ncols=self.vis_mosaic_cols, pad=2)
        #                 if mosaic is not None:
        #                     out_png = os.path.join(out_base, f"call{self._vis_counter:04d}_mosaic.png")
        #                     _save_heatmap2d(mosaic, out_png,
        #                                     title=f"layer{self.layer_id} call{self._vis_counter} mosaic",
        #                                     dpi=200, cmap="magma")
        #                 self._vis_counter += 1
        #         else:
        #             # 没有 grid 无法还原 3D
        #             pass
        # except Exception as e:
        #     print(f"[DiTBlock attn vis] skipped: {type(e).__name__}: {e}")


        x = self.gate(x, gate_msa, attn_out)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))

        if x_ref is not None:
            x_ref = self.gate(x_ref, gate_msa_ref, attn_out_ref)
            input_x_ref = modulate(self.norm2(x_ref), shift_mlp_ref, scale_mlp_ref)
            x_ref = self.gate(x_ref, gate_mlp_ref, self.ffn(input_x_ref))
        return x, x_ref


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        # print("self.has_pos_emb", self.has_pos_emb)
        # print(x.shape)
        # import pdb; pdb.set_trace()
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, t_mod):
        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
        x = self.gate(shift, 1 + scale, self.norm(x))
        x = self.head(x)
        # x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps, layer_idx)
            for layer_idx in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        self.has_image_pos_emb = has_image_pos_emb

    def patchify(self, x: torch.Tensor):
        # import pdb; pdb.set_trace()
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]
        x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
        return x, grid_size  # x, grid_size: (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
    
    # def _latent_mask_to_token_mask(self, mask_latent: torch.Tensor, patch_size: Tuple[int,int,int]) -> torch.Tensor:
    #     """
    #     mask_latent: [B, 1, F, H, W] 或 [B, F, H, W] ; 值域 {0,1} / [0,1]
    #     返回: [B, N] 的布尔 mask（N=f*h*w），1=允许注意力，0=禁止
    #     规则：token 内只要有任意像素为 1，则该 token 置 1（max_pool3d）
    #     """
    #     if mask_latent.dim() == 4:
    #         mask_latent = mask_latent.unsqueeze(1)  # [B,1,F,H,W]
    #     m = (mask_latent > 0.5).to(mask_latent.dtype)  # 二值化更稳
    #     pooled = F.max_pool3d(m, kernel_size=patch_size, stride=patch_size)  # [B,1,f,h,w]
    #     B, _, f, h, w = pooled.shape
    #     token_mask = pooled.reshape(B, 1, f*h*w).squeeze(1)  # [B, N]
    #     return (token_mask > 0.0)  # bool

    @torch.no_grad()
    def _latent_mask_to_token_mask(self, mask_lat: torch.Tensor, target_grid):
        """
        mask_lat: [B, C, F, H, W] 或 [B, 1, F, H, W] 或 [B, F, H, W]
        target_grid: (f, h, w) —— patchify 后的 token 网格尺寸
        return: [B, f*h*w]，1=允许注意力；0=屏蔽
        """
        # 统一到 [B, 1, F, H, W]
        if mask_lat.dim() == 4:               # [B, F, H, W]
            mask_lat = mask_lat.unsqueeze(1)
        assert mask_lat.dim() == 5, f"expect 4/5D mask, got {list(mask_lat.shape)}"

        B, C, F_lat, H_lat, W_lat = mask_lat.shape

        # —— 选择一种通道聚合方式 ——
        # 方案A：有任一通道为真就算“被遮罩”（常用于二值/稀疏掩码）
        # mask_1c = (mask_lat > 0).any(dim=1, keepdim=True).float()  # [B,1,F,H,W]
        # 方案B（可选）：取均值再阈值（适合连续值）：
        mask_1c = (mask_lat.float().mean(dim=1, keepdim=True) > 0.5).float()

        f_t, h_t, w_t = map(int, target_grid)
        pooled = F.adaptive_avg_pool3d(mask_1c, output_size=(f_t, h_t, w_t))  # [B,1,f_t,h_t,w_t]

        # 二值化（若想“只要覆盖一点点就算命中”，可用 >0）
        token_mask = (pooled > 0.5).flatten(1)  # [B, f_t*h_t*w_t]
        return token_mask.detach()


    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                latents_ref_img: Optional[torch.Tensor] = None,
                mask_input_for_v2v: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if latents_ref_img is not None:
            timestep_ref = torch.zeros_like(timestep)  # [B] with 0s
            t_ref = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep_ref))
            t_mod_ref = self.time_projection(t_ref).unflatten(1, (6, self.dim))
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)

        for blk in self.blocks:
            blk.self_attn.attn.grid_size = (f, h, w)
            blk.cross_attn.attn.grid_size = (f, h, w)

        for lid, blk in enumerate(self.blocks):
            blk.layer_id = lid                           # 便于命名
            blk.self_attn.attn.grid_main = (f, h, w)     # 主序列
            blk.cross_attn.attn.grid_main = (f, h, w)
            # if latents_ref_img is not None:
            #     blk.self_attn.attn.grid_ref = (f_ref, h_ref, w_ref)  # 若有参考

        
        offset = 0
        freqs = torch.cat([
            self.freqs[0][offset:f + offset].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][offset:h + offset].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][offset:w + offset].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

        freqs = [freqs]

        # import pdb; pdb.set_trace()
        ############################################################################################
        if latents_ref_img is not None:
            x_ref, (f_ref, h_ref, w_ref) = self.patchify(
                latents_ref_img
            )  # x_ref [1, 1024, 5120] [B, N, D]   f_ref = 1  h_ref = 32  w_ref = 32
            

            query_mask_ref_only = None
            if mask_input_for_v2v is not None:
                # 直接对齐到主序列网格 (f, h, w)
                query_mask_ref_only = self._latent_mask_to_token_mask(mask_input_for_v2v, (f, h, w))  # [B, f*h*w]  1=该行允许看ref

            # 下发到每个 block
            for blk in self.blocks:
                setattr(blk, "query_mask_ref_only", query_mask_ref_only)  # 只传 query 侧的mask
                setattr(blk, "key_token_mask_ref", None) 


            spanned_f = torch.linspace(0, f, self.num_heads+2, dtype=torch.long)[1:-1] + offset
            freqs_ref_f = self.freqs[0][spanned_f]
            freqs_ref_f = repeat(freqs_ref_f, 'nHead d -> nHead h_ref w_ref d',
                nHead=self.num_heads, h_ref=h_ref, w_ref=w_ref)
            
            freqs_ref_h = self.freqs[1][h + offset : h + offset + h_ref].view(1, h_ref, 1, -1).expand(self.num_heads, h_ref, w_ref, -1)
            freqs_ref_w = self.freqs[2][w + offset : w + offset + w_ref].view(1, 1, w_ref, -1).expand(self.num_heads, h_ref, w_ref, -1)

            freqs_ref = torch.cat([freqs_ref_f, freqs_ref_h, freqs_ref_w], dim=-1)
            freqs_ref = rearrange(freqs_ref, 'nHead h_ref w_ref d -> (h_ref w_ref) nHead d').to(x_ref.device)
        else:
            freqs_ref = None
        
        ### MOD: appending freqs_ref to the freqs list, rather than concatenating along the token dim
        if freqs_ref is not None:
            freqs.append(freqs_ref)
        ############################################################################################
    
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_ref = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            x_ref,
                            t_mod_ref,
                            use_reentrant=False,
                        )
                else:
                    x, x_ref = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        x_ref,
                        t_mod_ref,
                        use_reentrant=False,
                    )
            else:
                x, x_ref = block(x, context, t_mod, freqs, x_ref, t_mod_ref)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x

    def freeze_for_v2v(self):
        # Freeze all parameters
        for name, param in self.named_parameters():
            if 'patch_embedding' in name:
                param.requires_grad = False
    @staticmethod
    def state_dict_converter():
        return WanModelStateDictConverter()
    
    
class WanModelStateDictConverter:
    def __init__(self):
        pass

    def from_diffusers(self, state_dict):
        rename_dict = {
            "blocks.0.attn1.norm_k.weight": "blocks.0.self_attn.norm_k.weight",
            "blocks.0.attn1.norm_q.weight": "blocks.0.self_attn.norm_q.weight",
            "blocks.0.attn1.to_k.bias": "blocks.0.self_attn.k.bias",
            "blocks.0.attn1.to_k.weight": "blocks.0.self_attn.k.weight",
            "blocks.0.attn1.to_out.0.bias": "blocks.0.self_attn.o.bias",
            "blocks.0.attn1.to_out.0.weight": "blocks.0.self_attn.o.weight",
            "blocks.0.attn1.to_q.bias": "blocks.0.self_attn.q.bias",
            "blocks.0.attn1.to_q.weight": "blocks.0.self_attn.q.weight",
            "blocks.0.attn1.to_v.bias": "blocks.0.self_attn.v.bias",
            "blocks.0.attn1.to_v.weight": "blocks.0.self_attn.v.weight",
            "blocks.0.attn2.norm_k.weight": "blocks.0.cross_attn.norm_k.weight",
            "blocks.0.attn2.norm_q.weight": "blocks.0.cross_attn.norm_q.weight",
            "blocks.0.attn2.to_k.bias": "blocks.0.cross_attn.k.bias",
            "blocks.0.attn2.to_k.weight": "blocks.0.cross_attn.k.weight",
            "blocks.0.attn2.to_out.0.bias": "blocks.0.cross_attn.o.bias",
            "blocks.0.attn2.to_out.0.weight": "blocks.0.cross_attn.o.weight",
            "blocks.0.attn2.to_q.bias": "blocks.0.cross_attn.q.bias",
            "blocks.0.attn2.to_q.weight": "blocks.0.cross_attn.q.weight",
            "blocks.0.attn2.to_v.bias": "blocks.0.cross_attn.v.bias",
            "blocks.0.attn2.to_v.weight": "blocks.0.cross_attn.v.weight",
            "blocks.0.ffn.net.0.proj.bias": "blocks.0.ffn.0.bias",
            "blocks.0.ffn.net.0.proj.weight": "blocks.0.ffn.0.weight",
            "blocks.0.ffn.net.2.bias": "blocks.0.ffn.2.bias",
            "blocks.0.ffn.net.2.weight": "blocks.0.ffn.2.weight",
            "blocks.0.norm2.bias": "blocks.0.norm3.bias",
            "blocks.0.norm2.weight": "blocks.0.norm3.weight",
            "blocks.0.scale_shift_table": "blocks.0.modulation",
            "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
            "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
            "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
            "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
            "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
            "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
            "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
            "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
            "condition_embedder.time_proj.bias": "time_projection.1.bias",
            "condition_embedder.time_proj.weight": "time_projection.1.weight",
            "patch_embedding.bias": "patch_embedding.bias",
            "patch_embedding.weight": "patch_embedding.weight",
            "scale_shift_table": "head.modulation",
            "proj_out.bias": "head.head.bias",
            "proj_out.weight": "head.head.weight",
        }
        state_dict_ = {}
        for name, param in state_dict.items():
            if name in rename_dict:
                state_dict_[rename_dict[name]] = param
            else:
                name_ = ".".join(name.split(".")[:1] + ["0"] + name.split(".")[2:])
                if name_ in rename_dict:
                    name_ = rename_dict[name_]
                    name_ = ".".join(name_.split(".")[:1] + [name.split(".")[1]] + name_.split(".")[2:])
                    state_dict_[name_] = param
        if hash_state_dict_keys(state_dict) == "cb104773c6c2cb6df4f9529ad5c60d0b":
            config = {
                "model_type": "t2v",
                "patch_size": (1, 2, 2),
                "text_len": 512,
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "window_size": (-1, -1),
                "qk_norm": True,
                "cross_attn_norm": True,
                "eps": 1e-6,
            }
        else:
            config = {}
        return state_dict_, config
    
    def from_civitai(self, state_dict):
        if hash_state_dict_keys(state_dict) == "9269f8db9040a9d860eaca435be61814":
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 1536,
                "ffn_dim": 8960,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 12,
                "num_layers": 30,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "aafcfd9672c3a2456dc46e1cb6e52c70" or hash_state_dict_keys(state_dict) == "c1135057d34827d70116ccd4da5ca471":  ## this is 14b t2v
            config = {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 16,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        
        elif hash_state_dict_keys(state_dict) == "6bfcfb3b342cb286ce886889d519a77e":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6
            }
        elif hash_state_dict_keys(state_dict) == "3ef3b1f8e1dab83d5b71fd7b617f859f":
            config = {
                "has_image_input": True,
                "patch_size": [1, 2, 2],
                "in_dim": 36,
                "dim": 5120,
                "ffn_dim": 13824,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 16,
                "num_heads": 40,
                "num_layers": 40,
                "eps": 1e-6,
                "has_image_pos_emb": True
            }
        else:
            config = {}
        return state_dict, config