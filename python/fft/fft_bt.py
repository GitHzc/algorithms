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
def w(k, x):
   return cos(2 * pi * x / k) + 1j * sin(2 * pi * x / k)

'''
Task of multiprocessing, calculate 1 dimension fft or ifft.
'''
def task_bt(index, row, flags):
    start = time.time()
    M = len(row)
    d = M
    w = w1
    if flags == 1:
        d = 1
        w = w2
    result = []
    w_1_0 = w(1, 0)
    row = np.apply_along_axis(lambda a: a * w_1_0, 0, row)
    for u in range(M):
        r = np.copy(row)
        k = M // 2
        a = 2
        while k > 0:
            for x in range(k):
                for b in range(a//2, a):
                    r[a * x + b] *= w(2 * k, u)
            a *= 2
            k //= 2
        result.append((index, u,  sum(r) / d))
    end = time.time()
    print('Task %d run for %0.2f seconds' % (index, end - start))
    return result

'''
Calculating 1 dimension fft or ifft in the way from bottom to top.
If flags is 0, do fft.
If flags is 1, do ifft.
'''
def fft1d(input_img, flags):
    h = input_img.shape[0]
    w = input_img.shape[1]
    output_img = np.zeros((h,w), dtype=complex)
    p = Pool(7)
    result = []
    for i in range(h):
        row = np.array(input_img[i], dtype=complex)
        result.append(p.apply_async(task, args=(i, row, flags)))
    p.close()
    p.join()
    for res in result:
        row = res.get()
        for tup in row:
            output_img[tup[0]][tup[1]] = tup[2]
    return output_img

'''
Calculating 2 dimension fft or ifft.
If flags is 0, do fft.
If flags is 1, do ifft.
'''
def fft2d(input_img, flags):
    image_pad0 = pad0(input_img)
    h = image_pad0.shape[0]
    w = image_pad0.shape[1]
    output_img = np.zeros((h,w), dtype=complex)
    pass1 = fft1d(image_pad0, flags)
    pass2 = fft1d(pass1.T, flags)
    save_img = np.vectorize(abs)(pass2)
    cv2.imwrite('output_img_bt.png', save_img)
    return pass2

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
