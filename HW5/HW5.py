import copy
import numpy as np
from PIL import Image

def dilation(np_img, kernel):
    row = np_img.shape[0]
    col = np_img.shape[1]
    dilation_img = np.zeros((row, col), dtype=int)
    for i in range(row):
        for j in range(col):
            max_value = 0
            for k in kernel:
                tmp = 0
                new_i = i - k[0]
                new_j = j - k[1]
                if new_i >= 0 and new_i < row and \
                    new_j >= 0 and new_j < col:
                    tmp = np_img[new_i][new_j] + k[2]
                    if max_value < tmp:
                        max_value = tmp
            dilation_img[i][j] = max_value
    return dilation_img

def erosion(np_img, kernel):
    row = np_img.shape[0]
    col = np_img.shape[1]
    erosion_img = np.zeros((row, col), dtype=int)
    for i in range(row):
        for j in range(col):
            min_value = 256
            for k in kernel:
                tmp = 0
                new_i = i + k[0]
                new_j = j + k[1]
                if new_i >= 0 and new_i < row and \
                    new_j >= 0 and new_j < col:
                    tmp = np_img[new_i][new_j] - k[2]
                    if tmp < min_value:
                        min_value = tmp
            if min_value < 0:
                min_value = 0
            erosion_img[i][j] = min_value
    return erosion_img

def main():
    img = Image.open('lena.bmp')
    #img.show()
    np_img = np.array(img)
    print("HW5: ", np_img.shape)

    # kernel ([x, y, value])
    kernel = np.array(
        [
            [-2, -1,  0], [-2,  0,  0], [-2,  1,  0],
            [-1, -2,  0], [-1, -1,  0], [-1,  0,  0], [-1,  1,  0], [-1,  2,  0],
            [ 0, -2,  0], [ 0, -1,  0], [ 0,  0,  0], [ 0,  1,  0], [ 0,  2,  0],
            [ 1, -2,  0], [ 1, -1,  0], [ 1,  0,  0], [ 1,  1,  0], [ 1,  2,  0],
            [ 2, -1,  0], [ 2,  0,  0], [ 2,  1,  0]
        ]
    )

    # (a) Dilation
    np_img_a = dilation(np_img, kernel)
    img_a = Image.fromarray(np.uint8(np_img_a))
    img_a.save("result/Dilation.bmp")

    # (b) Erosion
    np_img_b = erosion(np_img, kernel)
    img_b = Image.fromarray(np.uint8(np_img_b))
    img_b.save("result/Erosion.bmp")

    # (c) Opening
    np_img_c = dilation(erosion(np_img, kernel), kernel)
    img_c = Image.fromarray(np.uint8(np_img_c))
    img_c.save("result/Opening.bmp")

    # (d) Closing
    np_img_d = erosion(dilation(np_img, kernel), kernel)
    img_d = Image.fromarray(np.uint8(np_img_d))
    img_d.save("result/Closing.bmp")

if __name__ == '__main__':
    main()