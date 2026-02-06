# batch_oc_eval.py  (GT vs Generated, per-frame inside bbox)
import os
import csv
import sys
import math
import glob
import argparse
import traceback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        help="母目录（包含 sample_*_generated.mp4 / sample_*_handpose.mp4 / sample_*_gt.mp4）",
    )
    parser.add_argument("--stride", type=int, default=1, help="评测帧间隔")
    parser.add_argument("--device", default=None, help="cuda 或 cpu；默认自动检测")
    parser.add_argument("--aggregate", default="mean", choices=["mean", "median"], help="分数聚合方式")
    parser.add_argument("--vis_subdir", default="oc_vis", help="可视化输出子目录名（位于 root 下）")
    parser.add_argument("--csv_name", default="oc_scores.csv", help="汇总结果 CSV 文件名（位于 root 下）")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    vis_dir = os.path.join(root, args.vis_subdir)
    os.makedirs(vis_dir, exist_ok=True)

    try:
        from oc_metric_with_viz import compute_oc_metric_with_viz
    except Exception as e:
        # Try import from current directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, script_dir)
        try:
            from oc_metric_with_viz import compute_oc_metric_with_viz
        except Exception as e2:
            print("请先把 OC 指标实现保存为 oc_metric_with_viz.py，并包含 compute_oc_metric_with_viz 函数")
            print("import 错误：", e2)
            sys.exit(1)

    def strip_suffix(name: str):
        for s in ["_generated.mp4", "_handpose.mp4", "_gt.mp4"]:
            if name.endswith(s):
                return name[: -len(s)]
        return None

    names = set()
    for pat in ("sample_*_generated.mp4", "sample_*_handpose.mp4", "sample_*_gt.mp4"):
        for f in glob.glob(os.path.join(root, pat)):
            base = os.path.basename(f)
            prefix = strip_suffix(base)
            if prefix:
                names.add(prefix)

    samples = sorted(names)
    if not samples:
        print(f"[WARN] 在 {root} 下未找到 sample_*_generated/handpose/gt.mp4")
        return

    csv_path = os.path.join(root, args.csv_name)
    overall_vals = []

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "sample",
            "video_generated",
            "bbox_mask_video",
            "gt_video_used",
            "overall",
            "n_frames",
            "n_valid",
            "vis_path",
            "error",
        ])

        for sample in samples:
            gen_path = os.path.join(root, f"{sample}_generated.mp4")
            mask_path = os.path.join(root, f"{sample}_handpose.mp4")   # 提供每帧 bbox 的视频
            gt_path   = os.path.join(root, f"{sample}_gt.mp4")
            out_vis_path = os.path.join(vis_dir, f"{sample}.mp4")

            missing = []
            if not os.path.exists(gen_path):  missing.append("generated")
            if not os.path.exists(mask_path): missing.append("bbox_mask")
            if not os.path.exists(gt_path):   missing.append("gt")

            if missing:
                writer.writerow([sample, gen_path, mask_path, gt_path, "", "", "", "", f"missing: {','.join(missing)}"])
                print(f"[skip] {sample}: 缺少 {missing}")
                continue

            try:
                # 新增：强制使用 ref_video_path 分支（逐帧对齐，在 bbox 内算 clip 相似度）
                res = None
                try:
                    # res = compute_oc_metric_with_viz(
                    #     video_path=gen_path,
                    #     bbox_mask_video_path=mask_path,
                    #     ref_video_path=gt_path,          # <<< 关键：传入 GT 视频
                    #     ref_mask_path=None,
                    #     output_vis_path=out_vis_path,
                    #     stride=args.stride,
                    #     device=args.device,
                    #     aggregate=args.aggregate,
                    #     mode="inside_bbox",               # 若实现支持，显式指明仅在 bbox 内计算
                    # )
                    res = compute_oc_metric_with_viz(
                        video_path=gen_path,
                        bbox_mask_video_path=mask_path,
                        ref_video_path=gt_path,          # ★ 逐帧 GT 模式
                        ref_mask_path=None,
                        output_vis_path=out_vis_path,
                        stride=args.stride,
                        device=args.device,
                        aggregate=args.aggregate,
                    )
                except TypeError as e:
                    raise RuntimeError(
                        "当前 oc_metric_with_viz.compute_oc_metric_with_viz 不支持参数 ref_video_path。\n"
                        "请按下方补丁更新 oc_metric_with_viz.py，使其支持 GT 视频逐帧评测。"
                    ) from e

                overall = res.get("overall", float("nan"))
                if isinstance(overall, (int, float)) and math.isfinite(float(overall)):
                    overall_vals.append(float(overall))

                writer.writerow([
                    sample,
                    gen_path,
                    mask_path,
                    gt_path,
                    (f"{overall:.6f}" if isinstance(overall, (int, float)) and math.isfinite(float(overall)) else ""),
                    res.get("n_frames", ""),
                    res.get("n_valid", ""),
                    res.get("vis_path", ""),
                    "",
                ])
                print(f"[OK] {sample}: overall={overall}")
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                writer.writerow([sample, gen_path, mask_path, gt_path, "", "", "", "", err])
                print(f"[ERR] {sample}: {err}")
                traceback.print_exc()

        if overall_vals:
            overall_mean = sum(overall_vals) / len(overall_vals)
            writer.writerow(["__OVERALL_MEAN__", "", "", "", f"{overall_mean:.6f}", "", "", "", ""])
            print(f"\nOverall mean across {len(overall_vals)} samples: {overall_mean:.6f}")
        else:
            writer.writerow(["__OVERALL_MEAN__", "", "", "", "", "", "", "", "no valid overall scores"])
            print("\nOverall mean: no valid overall scores")

    print(f"\nDone. 结果写入：{csv_path}\n可视化目录：{vis_dir}")

if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=7 python data_tools/batch_oc_eval.py --root "/root/paddlejob/workspace/huangxuan/wan_new/wan_object/0930/i2v_anchorcrafter_81f_rope_mochu_sft_wofirstframe_worefinbbox_e2_swap" --stride 1 --aggregate mean
