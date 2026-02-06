"""
Helpers for resizing with multiple CPU cores
"""
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F


dict_name_to_filter = {
    "PIL": {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX
    },
}


def build_resizer(mode):
    if mode == "clean":
        return make_resizer("PIL", False, "bicubic", (299, 299))
    elif mode == "legacy_tensorflow":
        return lambda x: x
    elif mode == "legacy_pytorch":
        return make_resizer("PyTorch", False, "bilinear", (299, 299))
    else:
        raise ValueError(f"Invalid mode {mode} specified")


def make_resizer(library, quantize_after, filter, output_size):
    if library == "PIL" and quantize_after:
        def func(x):
            x = Image.fromarray(x)
            x = x.resize(output_size, resample=dict_name_to_filter[library][filter])
            x = np.asarray(x).astype(np.uint8)
            return x

    elif library == "PIL" and not quantize_after:
        s1, s2 = output_size

        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=dict_name_to_filter[library][filter])
            return np.asarray(img).reshape(s1, s2, 1)

        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x

    elif library == "PyTorch":
        import warnings
        warnings.filterwarnings("ignore")

        def func(x):
            x = torch.Tensor(x.transpose((2, 0, 1)))[None, ...]
            x = F.interpolate(x, size=output_size, mode=filter, align_corners=False)
            x = x[0, ...].cpu().data.numpy().transpose((1, 2, 0)).clip(0, 255)
            if quantize_after:
                x = x.astype(np.uint8)
            return x
    else:
        raise NotImplementedError('library [%s] is not include' % library)
    return func