# import numpy as np
# import torch
# from torchvision.models import inception_v3
# from torchvision.transforms import ToTensor, Resize
# from scipy.linalg import sqrtm
# import cv2
# import glob
# import os
# import csv
# from decord import VideoReader, cpu
# from scipy import linalg
# from inception import InceptionV3
# import random
# from skimage.metrics import structural_similarity as SSIM


# def numerical_sort(file_name):
#     file_name = file_name.split('/')[-1]
#     num = int(file_name.split('___')[0])
#     return int(num)

# def get_activations(
#     files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
# ):
#     """Calculates the activations of the pool_3 layer for all images.

#     Params:
#     -- files       : List of image files paths
#     -- model       : Instance of inception model
#     -- batch_size  : Batch size of images for the model to process at once.
#                      Make sure that the number of samples is a multiple of
#                      the batch size, otherwise some samples are ignored. This
#                      behavior is retained to match the original FID score
#                      implementation.
#     -- dims        : Dimensionality of features returned by Inception
#     -- device      : Device to run calculations
#     -- num_workers : Number of parallel dataloader workers

#     Returns:
#     -- A numpy array of dimension (num images, dims) that contains the
#        activations of the given tensor when feeding inception with the
#        query tensor.
#     """
#     model.eval()
#     to_tensor_transform = ToTensor()

#     if batch_size > len(files):
#         print(
#             (
#                 "Warning: batch size is bigger than the data size. "
#                 "Setting batch size to data size"
#             )
#         )
#         batch_size = len(files)

#     ####files是F,512,512,3

#     pred_arr = np.empty((len(files), dims))

#     start_idx = 0
#     for i in range(0, len(files), batch_size):
#         batch = files[i:i+batch_size]
#         batch = torch.stack([to_tensor_transform(img) for img in batch]).to(device)
#         with torch.no_grad():
#             pred = model(batch)[0]
        
#         if pred.size(2) != 1 or pred.size(3) != 1:
#             pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

#         pred = pred.squeeze(3).squeeze(2).cpu().numpy()

#         pred_arr[start_idx : start_idx + pred.shape[0]] = pred
#         start_idx = start_idx + pred.shape[0]


#     return pred_arr

# def calculate_activation_statistics(
#     files, model, batch_size=50, dims=2048, device="cpu", num_workers=1
# ):
#     """Calculation of the statistics used by the FID.
#     Params:
#     -- files       : List of image files paths
#     -- model       : Instance of inception model
#     -- batch_size  : The images numpy array is split into batches with
#                      batch size batch_size. A reasonable batch size
#                      depends on the hardware.
#     -- dims        : Dimensionality of features returned by Inception
#     -- device      : Device to run calculations
#     -- num_workers : Number of parallel dataloader workers

#     Returns:
#     -- mu    : The mean over samples of the activations of the pool_3 layer of
#                the inception model.
#     -- sigma : The covariance matrix of the activations of the pool_3 layer of
#                the inception model.
#     """
#     act = get_activations(files, model, batch_size, dims, device, num_workers)
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma


# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance.
#     The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
#     and X_2 ~ N(mu_2, C_2) is
#             d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

#     Stable version by Dougal J. Sutherland.

#     Params:
#     -- mu1   : Numpy array containing the activations of a layer of the
#                inception net (like returned by the function 'get_predictions')
#                for generated samples.
#     -- mu2   : The sample mean over activations, precalculated on an
#                representative data set.
#     -- sigma1: The covariance matrix over activations for generated samples.
#     -- sigma2: The covariance matrix over activations, precalculated on an
#                representative data set.

#     Returns:
#     --   : The Frechet Distance.
#     """

#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)

#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     assert (
#         mu1.shape == mu2.shape
#     ), "Training and test mean vectors have different lengths"
#     assert (
#         sigma1.shape == sigma2.shape
#     ), "Training and test covariances have different dimensions"

#     diff = mu1 - mu2

#     # Product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         msg = (
#             "fid calculation produces singular product; "
#             "adding %s to diagonal of cov estimates"
#         ) % eps
#         print(msg)
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     # Numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError("Imaginary component {}".format(m))
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)

#     return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


# def calculate_fid(real_images, generated_images, model, batch_size=50, device="cuda"):
#     m1, s1 = calculate_activation_statistics(
#             real_images, model, batch_size, dims, device
#         )

#     m2, s2 = calculate_activation_statistics(
#             generated_images, model, batch_size, dims, device
#         )

#     fid_value = calculate_frechet_distance(m1, s1, m2, s2)

#     return fid_value

# def PSNR_new2(img1,img2):
# 	img1 = np.float64(img1) / (2**8-1)
# 	img2 = np.float64(img2) / (2**8-1)
# 	mse = np.mean(np.square(img1-img2))
# 	psnr = - 10 * np.log10(mse)
# 	# print(psnr)
# 	return psnr
    
# def PSNR(img1, img2):
#     mse = np.mean((img1-img2)**2)
#     if mse == 0:
#         return float('inf')
#     else:
#         return 20*np.log10(255/np.sqrt(mse))

# def PSNR_auto(img1, img2):
#     img1 = np.asarray(img1).astype(np.float32)
#     img2 = np.asarray(img2).astype(np.float32)
#     max_val = 1.0 if img1.max() <= 1.0 else 255.0
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * np.log10(max_val / np.sqrt(mse))


# dims = 2048
# device = "cuda:0"
# block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
# model = InceptionV3([block_idx]).to(device)
# model.eval()

# # Directory containing paired videos: sample_XXXX_generated.mp4 and sample_XXXX_gt.mp4
# video_dir = "/root/paddlejob/workspace/huangxuan/wan_new/wan_object/0729/test_metric"

# # Utility to sort numerically by the sample index
# def numerical_sort(value):
#     basename = os.path.basename(value)
#     idx_str = basename.split('_')[1]
#     return int(idx_str)

# # List and sort generated and ground-truth videos
# gen_videos = sorted(
#     glob.glob(os.path.join(video_dir, '*_generated.mp4')),
#     key=numerical_sort
# )
# gt_videos = sorted(
#     glob.glob(os.path.join(video_dir, '*_gt.mp4')),
#     key=numerical_sort
# )

# assert len(gen_videos) == len(gt_videos), \
#     f"Mismatch: {len(gt_videos)} GT videos vs {len(gen_videos)} generated videos"

# # Read frames and collect without cropping
# gt_imgs_list = []
# sy_imgs_list = []
# for gt_path, sy_path in zip(gt_videos, gen_videos):
#     gt_video = VideoReader(gt_path, ctx=cpu(0))
#     sy_video = VideoReader(sy_path, ctx=cpu(0))
    
#     # assert len(gt_video) == len(sy_video), \
#     #     f"Frame count mismatch in {gt_path} {len(gt_video)} vs {sy_path} {len(sy_video)} "

#     # frame_indices = list(range(len(gt_video)))

#     if len(gt_video) != len(sy_video) :
#         print(f"Frame count mismatch in {gt_path} {len(gt_video)} vs {sy_path} {len(sy_video)} ")
#     frame_indices = list(range(min(len(gt_video), len(sy_video), 81)))

#     gt_batch = gt_video.get_batch(frame_indices).asnumpy()
#     sy_batch = sy_video.get_batch(frame_indices).asnumpy()

#     gt_imgs_list.append(gt_batch)
#     sy_imgs_list.append(sy_batch)

# # Concatenate all videos' frames
# gt_imgs = np.concatenate(gt_imgs_list, axis=0)
# sy_imgs = np.concatenate(sy_imgs_list, axis=0)

# print("GT frames shape:", gt_imgs.shape)
# print("Generated frames shape:", sy_imgs.shape)

# # Compute FID
# batch_size = 50
# fid_score = calculate_fid(gt_imgs, sy_imgs, model, batch_size, device)
# print("FID score:", fid_score)

# # Compute PSNR and SSIM
# psnr_list = []
# psnr_auto_list = []

# ssim_list = []
# for i in range(gt_imgs.shape[0]):
#     gt_frame = gt_imgs[i:i+1]
#     sy_frame = sy_imgs[i:i+1]
#     psnr_list.append(PSNR(gt_frame, sy_frame))
#     psnr_auto_list.append(PSNR_auto(gt_frame, sy_frame))
#     ssim_list.append(SSIM(gt_frame[0], sy_frame[0], channel_axis=-1, data_range=255))

# psnr_score = np.mean(psnr_list)
# psnr_auto_score = np.mean(psnr_auto_list)
# ssim_score = np.mean(ssim_list)
# print("PSNR score:", psnr_score)
# print("PSNR_auto score:", psnr_auto_score)
# print("SSIM score:", ssim_score)


# # 结果输出路径（可改）
# results_save_path = os.path.join(video_dir, "evaluation_results_ori.csv")

# # 写入 CSV 文件
# with open(results_save_path, mode='w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["Metric", "Value"])
#     writer.writerow(["FID", fid_score])
#     writer.writerow(["PSNR", psnr_score])
#     writer.writerow(["PSNR_auto", psnr_auto_score])
#     writer.writerow(["SSIM", ssim_score])

# print(f"\n✅ Results saved to: {results_save_path}")


import os
import os
import sys
import glob
import csv
import torch
import numpy as np
import torch.nn.functional as F
import lpips
from torchvision.models.inception import inception_v3
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from PIL import Image
from scipy import linalg
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import imageio

# Add current directory to path for importing inception
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from inception import InceptionV3
from decord import VideoReader, cpu


def get_video_feature(video_frames, model, device):
    preprocess = Compose([
        Resize((299, 299)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    frames_tensor = torch.stack([preprocess(f).to(device) for f in video_frames])
    with torch.no_grad():
        features = model(frames_tensor)[0]  # [F, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [F, 2048]
    return features.mean(dim=0).cpu().numpy()


def load_video_frames(video_path):
    reader = imageio.get_reader(video_path)
    frames = [Image.fromarray(frame) for frame in reader]
    reader.close()
    return frames


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return float(fid)


def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    return ssim(img1, img2, channel_axis=-1, data_range=255)


def calculate_lpips(lpips_model, img1, img2, device):
    """Calculate LPIPS score between two images (numpy arrays HWC, 0-255)"""
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1.0
    return lpips_model(img1_tensor, img2_tensor).item()


def evaluate_metrics(gt_videos, pred_videos, device="cuda:0"):
    metrics = {}

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    fid_model = InceptionV3([block_idx]).to(device)
    fid_model.eval()

    gt_imgs_list, pred_imgs_list = [], []
    for gt_path, pred_path in zip(gt_videos, pred_videos):
        gt_video = VideoReader(gt_path, ctx=cpu(0))
        pred_video = VideoReader(pred_path, ctx=cpu(0))
        frame_indices = list(range(min(len(gt_video), len(pred_video), 81)))

        gt_batch = gt_video.get_batch(frame_indices).asnumpy()
        pred_batch = pred_video.get_batch(frame_indices).asnumpy()

        gt_imgs_list.append(gt_batch)
        pred_imgs_list.append(pred_batch)

    gt_imgs = np.concatenate(gt_imgs_list, axis=0)
    pred_imgs = np.concatenate(pred_imgs_list, axis=0)

    from torchvision.transforms.functional import to_tensor
    def get_activations(frames, model, batch_size=50):
        model.eval()
        pred_arr = np.empty((len(frames), dims))
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch = torch.stack([to_tensor(Image.fromarray(img)).to(device) for img in batch])
            with torch.no_grad():
                pred = model(batch)[0]
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[i : i + pred.shape[0]] = pred
        return pred_arr

    def calculate_activation_statistics(images):
        act = get_activations(images, fid_model)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    m1, s1 = calculate_activation_statistics(gt_imgs)
    m2, s2 = calculate_activation_statistics(pred_imgs)
    fid_img = calculate_frechet_distance(m1, s1, m2, s2)
    metrics["FID"] = fid_img

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    for i in range(gt_imgs.shape[0]):
        psnr_scores.append(calculate_psnr(gt_imgs[i], pred_imgs[i]))
        ssim_scores.append(calculate_ssim(gt_imgs[i], pred_imgs[i]))
        lpips_scores.append(calculate_lpips(lpips_model, gt_imgs[i], pred_imgs[i], device))

    metrics["PSNR"] = np.mean(psnr_scores)
    metrics["SSIM"] = np.mean(ssim_scores)
    metrics["LPIPS"] = np.mean(lpips_scores)

    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FID, FID-VID, PSNR, SSIM metrics for video generation")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing *_generated.mp4 and *_gt.mp4 files")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path (default: video_dir/evaluation_results.csv)")
    args = parser.parse_args()
    
    video_dir = args.video_dir
    device = args.device
    output_path = args.output or os.path.join(video_dir, "evaluation_results.csv")
    
    print(f"Video directory: {video_dir}")
    print(f"Device: {device}")
    print(f"Output: {output_path}")
    print("")
    
    pred_videos = sorted(glob.glob(os.path.join(video_dir, "*_generated.mp4")))
    gt_videos = sorted(glob.glob(os.path.join(video_dir, "*_gt.mp4")))
    
    if not pred_videos or not gt_videos:
        print("Error: No video files found!")
        print(f"  Generated videos: {len(pred_videos)}")
        print(f"  Ground truth videos: {len(gt_videos)}")
        sys.exit(1)
    
    print(f"Found {len(pred_videos)} generated videos and {len(gt_videos)} ground truth videos")
    print("")

    metrics = evaluate_metrics(gt_videos, pred_videos, device=device)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
    
    print(f"\n✅ Results saved to: {output_path}")
