import copy
import numpy as np
from PIL import Image

def part1():
    img = Image.open('lena.bmp')
    #img.show()
    np_img = np.array(img)
    print("Do part1: ", np_img.shape)
    row = np_img.shape[0]
    col = np_img.shape[1]

    # function (a): upside-down lena.bmp
    np_img_a = copy.deepcopy(np_img)
    for i in range(int(row / 2)):
        for j in range(col):
            tmp = np_img_a[i][j]
            np_img_a[i][j] = np_img_a[row - i - 1][j]
            np_img_a[row - i - 1][j] = tmp
    img_a = Image.fromarray(np_img_a)
    img_a.save("result/upside-down.bmp")


    # function (b): right-side-left lena.bmp
    np_img_b = copy.deepcopy(np_img)
    for i in range(row):
        for j in range(int(col/2)):
            tmp = np_img_b[i][j]
            np_img_b[i][j] = np_img_b[i][col - j -1]
            np_img_b[i][col - j -1] = tmp
    img_b = Image.fromarray(np_img_b)
    img_b.save("result/right-side-left.bmp")


    # function (c):  diagonally flip lena.bmp
    np_img_c = copy.deepcopy(np_img)
    for i in range(row):
        for j in range(i):
            tmp = np_img_c[i][j]
            np_img_c[i][j] = np_img_c[j][i]
            np_img_c[j][i] = tmp
    img_c = Image.fromarray(np_img_c)
    img_c.save("result/diagonally-flip.bmp")


def part2():
    img = Image.open('lena.bmp')
    #img.show()
    np_img = np.array(img)
    print("Do part2: ", np_img.shape)
    row = np_img.shape[0]
    col = np_img.shape[1]

    # function (d): rotate lena.bmp 45 degrees clockwise
    img_d = copy.deepcopy(img)
    angle = -45
    img_d = img_d.rotate(angle)
    img_d.save('result/rotate-45-degrees-clockwise.bmp')


    # function (e): shrink lena.bmp in half
    img_e = copy.deepcopy(img)
    img_e = img_e.resize((int(row/2), int(col/2)))
    img_e.save('result/shrink-in-half.bmp')


    # function (f): binarize lena.bmp at 128 to get a binary image
    np_img_f = copy.deepcopy(np_img)
    for i in range(row):
        for j in range(col):
            np_img_f[i][j] = 255 if(np_img_f[i][j] >= 128) else 0
    img_f = Image.fromarray(np_img_f)
    img_f.save("result/binarize-at-128-to-get-a-binary-image.bmp")


if __name__ == '__main__':
    part1()
    part2()