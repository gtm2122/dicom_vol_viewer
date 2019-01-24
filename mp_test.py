def a(x):
    return x*x

import numpy as np

def cb(args):
    #nonlocal sl
    #nonlocal im
    sl=args[1]
    im = args[0]
    #for slice_number in range(len(sl)):

    intercept = sl[0]
    slope = sl[1]
    padding =  sl[2]

    #img = im
    im[im==padding] = 0  
    #im[slice_number] = img
    # Shift 2 bits based difference 16 -> 14-bit as returned by jpeg2k_bit_depth
    #padded_pixels = np.where( img & (1 << 14))
    #image[slice_number] = np.right_shift( image[slice_number], 2)

    if slope != 1:
        im = slope * im.astype(np.float64)
        im = im.astype(np.int16)

    im += np.int16(intercept)
    #img[padded_pixels] = intercept

    return im