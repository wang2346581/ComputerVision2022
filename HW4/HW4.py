import copy
import numpy as np
from PIL import Image

def dilation(np_img, kernel):
    row = np_img.shape[0]
    col = np_img.shape[1]
    dilation_img = np.zeros((row, col), dtype=int)
    for i in range(row):
        for j in range(col):
            if np_img[i][j]:
                for k in kernel:
                    new_i = i + k[0]
                    new_j = j + k[1]
                    if new_i >= 0 and new_i < row and \
                        new_j >= 0 and new_j < col:
                        dilation_img[new_i][new_j] = 255
    return dilation_img

def erosion(np_img, kernel):
    row = np_img.shape[0]
    col = np_img.shape[1]
    erosion_img = np.zeros((row, col), dtype=int)
    for i in range(row):
        for j in range(col):
            contained = True
            for k in kernel:
                new_i = i + k[0]
                new_j = j + k[1]
                if new_i < 0 or new_i >= row or \
                    new_j < 0 or new_j >= col or \
                    not np_img[new_i][new_j]:
                        contained = False
                        break
            if contained == True:
                erosion_img[i][j] = 255
    return erosion_img

def get_complement(np_img):
    row = np_img.shape[0]
    col = np_img.shape[1]
    np_img_complement = np.zeros((row, col), dtype=int)
    for i in range(row):
        for j in range(col):
            np_img_complement[i][j] = 255 - np_img[i][j]
    return np_img_complement

def hit_and_miss(np_img, kernel_j, kernel_k):
    row = np_img.shape[0]
    col = np_img.shape[1]
    res = np.zeros((row, col), dtype=int)
    img_A  = copy.deepcopy(np_img)
    img_Ac = get_complement(img_A)
    first_half  = erosion(img_A, kernel_j)
    second_half = erosion(img_Ac, kernel_k)
    for i in range(row):
        for j in range(col):
            if first_half[i][j] and second_half[i][j]:
                res[i][j] = 255
    return res

def main():
    img = Image.open('lena.bmp')
    #img.show()
    np_img = np.array(img)
    print("HW4: ", np_img.shape)
    row = np_img.shape[0]
    col = np_img.shape[1]

    # a binary image (threshold at 128)
    for i in range(row):
        for j in range(col):
            np_img[i][j] = 255 if(np_img[i][j] >= 128) else 0

    # kernel ([x, y])
    kernel = np.array(
        [
            [-2, -1], [-2,  0], [-2,  1],
            [-1, -2], [-1, -1], [-1,  0], [-1,  1], [-1,  2],
            [ 0, -2], [ 0, -1], [ 0,  0], [ 0,  1], [ 0,  2],
            [ 1, -2], [ 1, -1], [ 1,  0], [ 1,  1], [ 1,  2],
            [ 2, -1], [ 2,  0], [ 2,  1]
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
    
    # (e) Hit-and-miss transform
    kernel_j = np.array(
        [
            [ 0, -1], [ 0,  0], [1,  0]
        ]
    )
    kernel_k = np.array(
        [
            [-1,  0], [-1,  1], [0,  1]
        ]
    )
    np_img_e = hit_and_miss(np_img, kernel_j, kernel_k)
    img_e = Image.fromarray(np.uint8(np_img_e))
    img_e.save("result/Hit-and-miss.bmp")

if __name__ == '__main__':
    main()