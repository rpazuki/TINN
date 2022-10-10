import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def pool2D(arr, window, stride=(1,1), op=np.average):
    ret = sliding_window_view(a,window)[::stride[0],::stride[1]]
    l = len(ret.shape)
    return op(ret, axis=tuple(range(l-2, l)))