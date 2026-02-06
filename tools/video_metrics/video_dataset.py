"""
Video dataset loaders for FVD metrics.
Extracted from DisCo's utils.py
"""

import re
import sys
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageSequence
import ffmpeg

# Add current directory to path for imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from resizer import make_resizer


# Supported image file extensions
EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
              'tif', 'tiff', 'webp', 'npy'}


def gif_to_nparray(gif_path):
    """Convert GIF file to numpy array of frames."""
    gif = Image.open(gif_path)
    frames = [np.array(frame.copy().convert('RGB'), dtype=np.uint8) 
              for frame in ImageSequence.Iterator(gif)]
    video = np.stack(frames)
    return video


class DatasetFVDVideoResize(torch.utils.data.Dataset):
    """
    Dataset for loading videos (mp4, gif) and resizing for FVD computation.
    
    Args:
        files: List of video file paths
        sample_duration: Number of frames per segment
        mode: 'FVD-3DInception' or 'MAE'
        img_size: Target size for resizing (e.g., 224)
        return_name: Whether to return video name along with data
    """

    def __init__(self, files, sample_duration=16, mode='FVD-3DInception', 
                 img_size=224, return_name=False):
        self.files = files
        self.pixel_mean = torch.as_tensor(np.array([114.7748, 107.7354, 99.4750]))
        self.img_size = img_size
        self.sample_duration = sample_duration
        self.mode = mode
        self.resize_func = make_resizer("PIL", False, "bicubic", (img_size, img_size))
        self.return_name = return_name
 
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            path = str(self.files[i])

            # Load video based on format
            if Path(path).suffix == '.gif':
                video = gif_to_nparray(path)
            else:
                # Use ffmpeg to decode video
                probe = ffmpeg.probe(path)
                video_stream = next((stream for stream in probe['streams'] 
                                   if stream['codec_type'] == 'video'), None)
                width = int(video_stream['width'])
                height = int(video_stream['height'])
                out, _ = (ffmpeg.input(path)
                         .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                         .run(capture_stdout=True, quiet=True))
                video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

            # Resize each frame
            video_resize = []
            for vim in video:
                vim_resize = self.resize_func(vim)
                video_resize.append(vim_resize)

            video = np.stack(video_resize, axis=0)
            video = torch.as_tensor(video.copy()).float()
            num_v = video.shape[0]

            num_seg = num_v // self.sample_duration

            # Normalize based on model type
            if self.mode == "FVD-3DInception" or self.mode == 'MAE':
                video = video / 127.5 - 1
                video = video.unsqueeze(0).permute(0, 4, 1, 2, 3).float()

            if self.return_name:
                return video, Path(path).stem
            return video
            
        except Exception as e:
            print(f'{i} skipped because {e}')
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))


class DatasetFVDVideoFromFramesResize(torch.utils.data.Dataset):
    """
    Dataset for loading video from frame sequences (images) for FVD computation.
    
    This is used when videos are stored as sequences of image files instead of
    video files (mp4/gif).
    
    Args:
        files: List of image file paths (first frame of each sequence)
        sample_duration: Number of frames per video clip
        mode: 'FVD-3DInception' or 'MAE'
        img_size: Target size for resizing
        return_name: Whether to return video name along with data
    """

    def __init__(self, files, sample_duration=16, mode='FVD-3DInception', 
                 img_size=224, return_name=False):
        # Frame naming patterns to match
        frame_format1 = r"^(TiktokDance_\d+_)(\d+)(\D*\.\w+)$"
        frame_format2 = r"^(TiktokDance_\d+_\d+_1x1_)(\d+)(\D*\.\w+)$"
        frame_format3 = r"^(\d+_\d+_1x1__)(\d+)(.\D*\.\w+)$"
        frame_format4 = r"^(S\d{3}C\d{3}P\d{3}R\d{3}A\d{3}_)(\d{4})(\D*\.\w+)$"

        files = sorted(files)
        self.video_frames = {}
        
        # Group frames into video sequences
        for file in files:
            file_name = Path(file).name
            file_parent = Path(file).parent
            
            # Try to match frame naming pattern
            folder_format_re = None
            for pattern in [frame_format1, frame_format2, frame_format3, frame_format4]:
                if re.match(pattern, file_name):
                    folder_format_re = pattern
                    break
            
            if folder_format_re is None:
                print(f"Frame name '{file_name}' does not match any format")
                continue
        
            match = re.match(folder_format_re, file_name)
            frame_index = int(match.group(2))

            # Find consecutive frames
            self.video_frames[file_name] = [file]
            for i in range(1, sample_duration):
                next_frame_file_name = (match.group(1) + 
                                      str(frame_index + i).zfill(len(match.group(2))) + 
                                      match.group(3))
                next_frame_file_path = Path(file_parent, next_frame_file_name)
                if not next_frame_file_path.exists():
                    del self.video_frames[file_name]
                    break
                self.video_frames[file_name].append(next_frame_file_path.as_posix())

        self.pixel_mean = torch.as_tensor(np.array([114.7748, 107.7354, 99.4750]))
        self.img_size = img_size
        self.sample_duration = sample_duration
        self.mode = mode
        self.return_name = return_name
        self.resize_func = make_resizer("PIL", False, "bicubic", (img_size, img_size))
 
    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, i):
        try:
            video_name = list(self.video_frames.keys())[i]
            frame_list = []
            
            # Load all frames in sequence
            for frame_path in self.video_frames[video_name]:
                frame = Image.open(frame_path).convert('RGB')
                frame_list.append(np.array(frame))
            video = np.stack(frame_list, axis=0)

            # Resize each frame
            video_resize = []
            for vim in video:
                vim_resize = self.resize_func(vim)
                video_resize.append(vim_resize)

            video = np.stack(video_resize, axis=0)
            video = torch.as_tensor(video.copy()).float()
            num_v = video.shape[0]

            num_seg = num_v // self.sample_duration

            # Normalize
            if self.mode == "FVD-3DInception" or self.mode == 'MAE':
                video = video / 127.5 - 1
                video = video.unsqueeze(0).permute(0, 4, 1, 2, 3).float()

            if self.return_name:
                return video, Path(video_name).stem
            return video
            
        except Exception as e:
            print(f'{i} skipped because {e}')
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))