"""
Image resizing utilities for video metrics.
Extracted from DisCo's resize.py
"""

import numpy as np
from PIL import Image

# Mapping of filter names to PIL constants
dict_name_to_filter = {
    "PIL": {
        "bicubic": Image.BICUBIC,
        "bilinear": Image.BILINEAR,
        "nearest": Image.NEAREST,
        "lanczos": Image.LANCZOS,
        "box": Image.BOX
    },
}


def make_resizer(library, quantize_after, filter_type, output_size):
    """
    Construct a function that resizes a numpy image.
    
    Args:
        library: str, only 'PIL' is supported in this version
        quantize_after: bool, whether to quantize to uint8 after resizing
        filter_type: str, interpolation method (e.g., 'bicubic', 'bilinear')
        output_size: tuple, (height, width) of output image
    
    Returns:
        A function that takes a numpy array [H, W, C] and returns resized array
    """
    if library == "PIL" and quantize_after:
        def func(x):
            x = Image.fromarray(x)
            x = x.resize(output_size, resample=dict_name_to_filter[library][filter_type])
            x = np.asarray(x).astype(np.uint8)
            return x

    elif library == "PIL" and not quantize_after:
        s1, s2 = output_size

        def resize_single_channel(x_np):
            img = Image.fromarray(x_np.astype(np.float32), mode='F')
            img = img.resize(output_size, resample=dict_name_to_filter[library][filter_type])
            return np.asarray(img).reshape(s1, s2, 1)

        def func(x):
            x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
            x = np.concatenate(x, axis=2).astype(np.float32)
            return x

    else:
        raise NotImplementedError(f'library [{library}] with quantize_after={quantize_after} is not implemented')
    
    return func