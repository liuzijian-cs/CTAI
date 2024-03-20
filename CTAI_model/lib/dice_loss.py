import numpy as np
def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.

    这个函数计算的是Dice系数（也称为Sørensen–Dice系数），这是一种用于衡量两个集合相似度的统计工具，常用于比较样本、图像或文本的相似性。
    它特别在医学图像分析中用于评估图像分割的效果。Dice系数的值范围从 0 到 1 ，其中1表示完全相似，0表示无相似性。
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool_)
    im2 = np.asarray(im2).astype(np.bool_)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # 俩都为全黑
    if not (im1.any() or im2.any()):
        return 1.0

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    res = 2. * intersection.sum() / (im1.sum() + im2.sum())
    return np.round(res, 5)