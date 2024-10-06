import numpy as np


def neglog_normalize(_img):
    assert _img.ndim == 2
    if np.all(_img == 0):
        print("input image is zero... this is likely a mistake?")
        return np.zeros_like(_img, dtype=np.uint8)
    margin = _img.shape[0] // 8
    # preprocess only inner region (1/4 image margin) (avoids pixel-errors at boarder)
    _img = _img.astype(np.float64)
    ROI = _img[margin:-margin, margin:-margin]
    cval = np.median(ROI) + 3 * np.std(ROI)  # set contrast window upper limit to median + 3*std
    if cval == 0:
        print("cval is zero... this is likely a mistake?")
        return np.zeros_like(_img, dtype=np.uint8)
    _img = np.minimum(_img, cval) / cval
    # apply neglog and cast to uint8
    _img = -np.log(np.maximum(_img, np.finfo(dtype=_img.dtype).eps))
    ROI = _img[margin:-margin, margin:-margin]
    _img = (_img - ROI.min()) / (ROI.max() - ROI.min())
    _img = (np.clip(_img * 255, 0, 255)).astype(np.uint8)
    return _img