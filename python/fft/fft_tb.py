import cv2
import sys
import numpy as np
import time
from math import log, ceil, cos, sin, exp, pi
from multiprocessing import Pool

'''
Padding the input image with zero so its height and width are power of 2.
'''
def pad0(input_img):
    src_h = input_img.shape[0]
    src_w = input_img.shape[1]
    exp_h = ceil(log(src_h, 2))
    exp_w = ceil(log(src_w, 2))
    new_h = 2**exp_h
    new_w = 2**exp_w
    print('src_h:%d, src_w:%d, new_h:%d, new_w:%d' % (src_h, src_w, new_h, new_w))
    pad_r = np.zeros((src_h, new_w - src_w), dtype=int)
    pad_b = np.zeros((new_h - src_h, new_w), dtype=int)
    output_img = np.concatenate((input_img, pad_r), axis=1)
    output_img = np.concatenate((output_img, pad_b), axis=0)
    print('output_image: %d %d' % (output_img.shape[0], output_img.shape[1]))
    return output_img

'''
Calculating exp(-j2πx/k).
'''
def w1(k, x):
    return cos(2 * pi * x / k) + 1j * sin(2 * pi * x / k)

'''
Calculating exp(j2πx/k)
'''
def w2(k, x):
    return cos(2 * pi * x / k) - 1j * sin(2 * pi * x / k)

'''
Calculating 1 dimension fft or ifft in the way of recursion from top to bottom.
If flags is 0, do fft.
If flags is 1, do ifft.
'''
def fft1d(u, M, arr, flags):
    if M == 1:
        return arr[0] * w1(1, 0)
    M = M // 2
    arr_e = [arr[2 * i] for i in range(M)]
    arr_o = [arr[2 * i + 1] for i in range(M)]
    if flags == 0:
        return 0.5 * (fft1d(u, M, arr_e, flags) + fft1d(u, M, arr_o, flags) * w1(2 * M, u))
    else:
        return fft1d(u, M, arr_e, flags) + fft1d(u, M, arr_o, flags) * w2(2 * M, u)

'''
Task of multiprocessing, calculate 1 dimension fft or ifft.
'''
def task(index, M, arr, flags):
    start = time.time()
    res = []
    for u in range(M):
        res.append((index, u, fft1d(u, M, arr, flags)))
    end = time.time()
    print('Task %d run for %0.2f seconds' % (index, end - start))
    return res

'''
Calculating 2 dimension fft or ifft.
If flags is 0, do fft.
If flags is 1, do ifft.
'''
def fft2d(input_img, flags):
    image_pad0 = pad0(input_img)
    h = image_pad0.shape[0]
    w = image_pad0.shape[1]
    temp = np.zeros((h,w), dtype=complex)
    output_img = np.zeros((h,w), dtype=complex)

    p1 = Pool(7)
    result = []

    for i in range(h):
        row = image_pad0[i]
        result.append(p1.apply_async(task, args=(i, w, row, flags)))

    p1.close()
    p1.join()
    for res in result:
        row = res.get()
        for tup in row:
            temp[tup[0]][tup[1]] = tup[2]

    print('start processing horizontally')
    p2 = Pool(7)
    result = []
    for j in range(w):
        col = temp[:,j]
        result.append(p2.apply_async(task, args=(j, h, col, flags)))

    p2.close()
    p2.join()
    for res in result:
        col = res.get()
        for tup in col:
            output_img[tup[1]][tup[0]] = tup[2]

    save_img = np.vectorize(abs)(output_img)
    cv2.imwrite('output_img.png', save_img)
    return output_img

'''
def main(img_path, flags):
    input_img = cv2.imread(img_path, 0)
    fft2d(input_img, flags)
    output_img = cv2.imread('output_img.png', 0)
    cv2.imshow('output_img.png', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == __main__:
    img_path = sys.argv[1]
    flags = int(sys.argv[2])
    main(img_path, flags)
'''
