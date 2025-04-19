# -*- coding: utf-8 -*-
# @Author  : xuelun

import cv2
import torch
import numpy as np


# ------------
# DATA TOOLS
# ------------
def imread_gray(path, augment_fn=None):
    if augment_fn is None:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)


def imread_color(path, augment_fn=None):
    if augment_fn is None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new = max((w // df), 1) * df
        h_new = max((h // df), 1) * df
        # resize = int(max(max(w, h) // df, 1) * df)
        # w_new, h_new = get_resized_wh(w, h, resize)
        # scale = resize / x
        # w_new, h_new = map(lambda x: int(max(x // df, 1) * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
    elif inp.ndim == 3:
        padded = np.zeros((pad_size, pad_size, inp.shape[-1]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
    else:
        raise NotImplementedError()

    if ret_mask:
        mask = np.zeros((pad_size, pad_size), dtype=bool)
        mask[:inp.shape[0], :inp.shape[1]] = True

    return padded, mask


def split(n, k):
    d, r = divmod(n, k)
    return [d + 1] * r + [d] * (k - r)


def read_images(path, max_resize, df, padding, augment_fn=None, image=None):
    """
    Args:
        path: string
        max_resize (int): max image size after resied
        df (int, optional): image size division factor.
                            NOTE: this will change the final image size after img_resize
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
        image: RGB image
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    assert max_resize is not None

    image = imread_color(path, augment_fn) if image is None else image # (w,h,3) image is RGB
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # resize image
    w, h = image.shape[1], image.shape[0]
    if max(w, h) > max_resize:
        w_new, h_new = get_resized_wh(w, h, max_resize) # make max(w, h) to max_size
    else:
        w_new, h_new = w, h

    w_new, h_new = get_divisible_wh(w_new, h_new, df) # make image divided by df and must <= max_size
    image = cv2.resize(image, (w_new, h_new))  # (w',h',3)
    gray = cv2.resize(gray, (w_new, h_new))  # (w',h',3)
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)

    # padding
    mask = None
    if padding:
        image, _ = pad_bottom_right(image, max_resize, ret_mask=False)
        gray, mask = pad_bottom_right(gray, max_resize, ret_mask=True)
        mask = torch.from_numpy(mask)

    gray = torch.from_numpy(gray).float()[None] / 255 # (1,h,w)
    image = torch.from_numpy(image).float() / 255  # (h,w,3)
    image = image.permute(2,0,1) # (3,h,w)

    resize = [h_new, w_new]

    return gray, image, scale, resize, mask

def read_images_two_sizes(path, max_resize, fine_resize, df, padding, augment_fn=None, image=None):
    """
    Args:
        path: string
        max_resize (int): max image size after resized
        fine_resize (int): max image size for mast3r coarse-to-fine matching
        df (int, optional): image size division factor.
                            NOTE: this will change the final image size after img_resize
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
        image: RGB image
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    assert max_resize is not None

    image = imread_color(path, augment_fn) if image is None else image # (w,h,3) image is RGB
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # resize image
    w, h = image.shape[1], image.shape[0]
    if max(w, h) > max_resize:
        w_new, h_new = get_resized_wh(w, h, max_resize) # make max(w, h) to max_size
    else:
        w_new, h_new = w, h

    w_new, h_new = get_divisible_wh(w_new, h_new, df) # make image divided by df and must <= max_size
    image1 = cv2.resize(image, (w_new, h_new))  # (w',h',3)
    gray1 = cv2.resize(gray, (w_new, h_new))  # (w',h',3)
    scale1 = torch.tensor([w / w_new, h / h_new], dtype=torch.float)
    # padding
    mask1 = None
    if padding:
        image1, _ = pad_bottom_right(image1, max_resize, ret_mask=False)
        gray1, mask1 = pad_bottom_right(gray1, max_resize, ret_mask=True)
        mask1 = torch.from_numpy(mask1)
    gray1 = torch.from_numpy(gray1).float()[None] / 255 # (1,h,w)
    image1 = torch.from_numpy(image1).float() / 255  # (h,w,3)
    image1 = image1.permute(2,0,1) # (3,h,w)

    if max(w, h) > fine_resize:
        w_fine, h_fine = get_resized_wh(w, h, fine_resize) # make max(w, h) to max_size
    else:
        w_fine, h_fine = w, h
    w_fine, h_fine = get_divisible_wh(w_fine, h_fine, df) # make image divided by df and must <= max_size
    image2 = cv2.resize(image, (w_fine, h_fine))  # (w',h',3)
    # padding
    if padding:
        image2, _ = pad_bottom_right(image2, fine_resize, ret_mask=False)
       
    image2 = torch.from_numpy(image2).float() / 255  # (h_fine,w_fine,3)

    resize = [h_new, w_new]

    return gray1, image1, scale1, resize, mask1, image2