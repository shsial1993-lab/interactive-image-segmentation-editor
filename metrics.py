"""Binary segmentation evaluation. Undefined denominators return None."""
import numpy as np
from PIL import Image, ImageOps


def load_reference(image, shape):
    image = ImageOps.exif_transpose(image)
    a = np.asarray(image)
    if a.ndim == 3:
        if a.shape[2] == 4 and not np.all(a[..., 3] == 255):
            raise ValueError('Use an opaque binary mask, not a transparent cutout.')
        if not (np.array_equal(a[..., 0], a[..., 1]) and np.array_equal(a[..., 1], a[..., 2])):
            raise ValueError('Mask must be grayscale: 0/1 or 0/255. Color photographs are not reference masks.')
        a = a[..., 0]
    if a.ndim != 2 or not (set(np.unique(a)) <= {0, 1} or set(np.unique(a)) <= {0, 255}):
        raise ValueError('Use a binary PNG mask with values 0/1 or 0/255; no JPEG or soft edges.')
    if a.shape != shape:
        raise ValueError(f'Mask dimensions {a.shape[::-1]} must match working image {shape[::-1]}. Resize with nearest-neighbor before upload; no automatic alignment is assumed.')
    return a > 0


def evaluate(pred, truth):
    p, t = np.asarray(pred, bool), np.asarray(truth, bool)
    if p.shape != t.shape or p.size == 0:
        raise ValueError('Masks must have identical nonempty dimensions.')
    tp = int(np.sum(p & t)); fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t)); tn = int(np.sum(~p & ~t))
    ratio = lambda n, d: n / d if d else None
    fg = ratio(tp, tp + fp + fn)
    bg = ratio(tn, tn + fp + fn)
    valid = [x for x in (fg, bg) if x is not None]
    return {'Foreground IoU': fg, 'Background IoU': bg,
            'mIoU (foreground + background)': float(np.mean(valid)),
            'Foreground Dice': ratio(2 * tp, 2 * tp + fp + fn),
            'Foreground precision': ratio(tp, tp + fp),
            'Foreground recall': ratio(tp, tp + fn),
            'Pixel accuracy': (tp + tn) / p.size,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}


def error_map(pred, truth):
    out = np.zeros((*pred.shape, 3), np.uint8)
    out[pred & truth] = [40, 190, 90]
    out[pred & ~truth] = [240, 60, 60]
    out[~pred & truth] = [50, 110, 255]
    return out
