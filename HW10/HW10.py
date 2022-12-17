import numpy as np
from PIL import Image

def getZeroCrossingKernel(size = 1):
    zKernel = []
    rStart, rEnd = -size, size + 1
    for i in range(rStart, rEnd):
        for j in range(rStart, rEnd):
            if i == 0 and j == 0:
                continue
            zKernel.append([i, j])
    return np.array(zKernel)

def getPadding(img):
    row, col = img.shape
    resImg = np.zeros((row + 2, col + 2), dtype=int)
    resRow, resCol = resImg.shape
    resImg[0][0] = img[0][0] # up, left
    resImg[0][resCol - 1] = img[0][col - 1] # up, right
    resImg[resRow - 1][0] = img[row - 1][0] # bottom, left
    resImg[resRow - 1][resCol - 1] = img[row - 1][col - 1] # bottom, right
    for j in range(1, resCol - 1): # up and bottom
        resImg[0][j] = img[0][j - 1]
        resImg[resRow - 1][j] = img[row - 1][j - 1]
    for i in range(1, resRow - 1): # left and right
        resImg[i][0] = img[i - 1][0]
        resImg[i][resCol - 1] = img[i - 1][col - 1]
    for i in range(1, resRow - 1): # Remain Pixel
        for j in range(1, resCol - 1):
            resImg[i][j] = img[i - 1][j - 1]
    return resImg

def getLaplacian(img, threshold, kernel, kScale = 1, kSize = 3):
    # kScale for erasing the rounding problem in Python
    kShift = kSize // 2
    kPadding = kShift * 2
    # generate coordination of 8-neighborhood: x, y
    zKernel = getZeroCrossingKernel()
    row, col = img.shape
    resImg = np.zeros((row - kPadding, col - kPadding), dtype=int)
    laplacianMask = np.zeros((row - kPadding, col - kPadding), dtype=int)
    resRow, resCol = resImg.shape
    for i in range(resRow):
        for j in range(resCol):
            gradientMagnitude = 0.0
            for k in kernel:
                gradientMagnitude += img[i + kShift + int(k[0])][j + kShift + int(k[1])] * k[2]
            if gradientMagnitude >= threshold * kScale:
                laplacianMask[i][j] = 1
            elif gradientMagnitude <= -threshold * kScale:
                laplacianMask[i][j] = -1
            else:
                laplacianMask[i][j] = 0

    for i in range(kShift):
        laplacianMask = getPadding(laplacianMask)

    for i in range(resRow):
        for j in range(resCol):
            hasEdge = False
            if laplacianMask[i + kShift][j + kShift] == 1:
                for z in zKernel:
                    if laplacianMask[i + kShift + int(z[0])][j + kShift + int(z[1])] == -1:
                        hasEdge = True
                        break
            if not hasEdge:
                resImg[i][j] = 255
    return resImg

def getLaplacianCommon1(img, threshold = 15):
    # kernel: x, y, value
    kernel = np.array([
        [-1, -1, 0], [-1, 0,  1], [-1, 1, 0],
        [ 0, -1, 1], [ 0, 0, -4], [ 0, 1, 1],
        [ 1, -1, 0], [ 1, 0,  1], [ 1, 1, 0],
    ])
    return getLaplacian(img, threshold, kernel)

def getLaplacianCommon2(img, threshold = 15):
    # kScale for erasing the rounding problem in Python
    # kernel: x, y, value
    kernel = np.array([
        [-1, -1, 1], [-1, 0,  1], [-1, 1, 1],
        [ 0, -1, 1], [ 0, 0, -8], [ 0, 1, 1],
        [ 1, -1, 1], [ 1, 0,  1], [ 1, 1, 1],
    ])
    return getLaplacian(img, threshold, kernel, kScale = 3)

def getMinVarLaplacian(img, threshold = 20):
    # kScale for erasing the rounding problem in Python
    # kernel: x, y, value
    kernel = np.array([
        [-1, -1,  2], [-1, 0, -1], [-1, 1,  2],
        [ 0, -1, -1], [ 0, 0, -4], [ 0, 1, -1],
        [ 1, -1,  2], [ 1, 0, -1], [ 1, 1,  2],
    ])
    return getLaplacian(img, threshold, kernel, kScale = 3)

def getLaplacianOfGaussian(img, threshold = 3000):
    # kernel: x, y, value
    kernel1 = [
        [-5, -5,  0], [-5, -4,  0], [-5, -3,  0], [-5, -2, -1], [-5, -1, -1],
        [-5,  5,  0], [-5,  4,  0], [-5,  3,  0], [-5,  2, -1], [-5,  1, -1],
        [-5,  0, -2],

        [-4, -5,  0], [-4, -4,  0], [-4, -3, -2], [-4, -2, -4], [-4, -1, -8],
        [-4,  5,  0], [-4,  4,  0], [-4,  3, -2], [-4,  2, -4], [-4,  1, -8],
        [-4,  0, -9],

        [-3, -5,  0], [-3, -4, -2], [-3, -3, -7], [-3, -2, -15], [-3, -1, -22],
        [-3,  5,  0], [-3,  4, -2], [-3,  3, -7], [-3,  2, -15], [-3,  1, -22],
        [-3,  0, -23],

        [-2, -5, -1], [-2, -4, -4], [-2, -3, -15], [-2, -2, -24], [-2, -1, -14],
        [-2,  5, -1], [-2,  4, -4], [-2,  3, -15], [-2,  2, -24], [-2,  1, -14],
        [-2,  0, -1],

        [-1, -5, -1], [-1, -4, -8], [-1, -3, -22], [-1, -2, -14], [-1, -1,  52],
        [-1,  5, -1], [-1,  4, -8], [-1,  3, -22], [-1,  2, -14], [-1,  1,  52],
        [-1,  0, 103],

        [ 0, -5, -2], [ 0, -4, -9], [ 0, -3, -23], [ 0, -2, -1], [ 0, -1,  103],
        [ 0,  5, -2], [ 0,  4, -9], [ 0,  3, -23], [ 0,  2, -1], [ 0,  1,  103],
        [ 0,  0, 178],
    ]
    kernel2 = []
    for k in kernel1:
        if k[0] < 0:
            kernel2.append([-k[0], k[1], k[2]])
    kernel = np.array(kernel1 + kernel2)
    return getLaplacian(img, threshold, kernel, kSize = 11)

def getDifferenceOfGaussian(img, threshold = 1):
    # kernel: x, y, value
    kernel1 = [
        [-5, -5, -1], [-5, -4, -3], [-5, -3, -4], [-5, -2, -6], [-5, -1, -7],
        [-5,  5, -1], [-5,  4, -3], [-5,  3, -4], [-5,  2, -6], [-5,  1, -7],
        [-5,  0, -8],

        [-4, -5, -3], [-4, -4, -5], [-4, -3, -8], [-4, -2, -11], [-4, -1, -13],
        [-4,  5, -3], [-4,  4, -5], [-4,  3, -8], [-4,  2, -11], [-4,  1, -13],
        [-4,  0, -13],

        [-3, -5, -4], [-3, -4, -8], [-3, -3, -12], [-3, -2, -16], [-3, -1, -17],
        [-3,  5, -4], [-3,  4, -8], [-3,  3, -12], [-3,  2, -16], [-3,  1, -17],
        [-3,  0, -17],

        [-2, -5, -6], [-2, -4, -11], [-2, -3, -16], [-2, -2, -16], [-2, -1,  0],
        [-2,  5, -6], [-2,  4, -11], [-2,  3, -16], [-2,  2, -16], [-2,  1,  0],
        [-2,  0, 15],

        [-1, -5, -7], [-1, -4, -13], [-1, -3, -17], [-1, -2,  0], [-1, -1,  85],
        [-1,  5, -7], [-1,  4, -13], [-1,  3, -17], [-1,  2,  0], [-1,  1,  85],
        [-1,  0, 160],

        [ 0, -5, -8], [ 0, -4, -13], [ 0, -3, -17], [ 0, -2, 15], [ 0, -1,  160],
        [ 0,  5, -8], [ 0,  4, -13], [ 0,  3, -17], [ 0,  2, 15], [ 0,  1,  160],
        [ 0,  0, 283],
    ]
    kernel2 = []
    for k in kernel1:
        if k[0] < 0:
            kernel2.append([-k[0], k[1], k[2]])
    kernel = np.array(kernel1 + kernel2)
    return getLaplacian(img, threshold, kernel, kSize = 11)

def main():
    img = Image.open('lena.bmp')
    np_img = np.array(img, dtype=np.int)
    print("HW10: ", np_img.shape)
    np_img_padding1 = getPadding((np_img))

    #(a) Laplace Mask1 (0, 1, 0, 1, -4, 1, 0, 1, 0): 15
    np_a = getLaplacianCommon1(np_img_padding1, threshold = 15)
    Image.fromarray(np.uint8(np_a)).save('result/LaplaceMask1.bmp')

    #(b) Laplace Mask2 (1, 1, 1, 1, -8, 1, 1, 1, 1)
    np_b = getLaplacianCommon2(np_img_padding1, threshold = 15)
    Image.fromarray(np.uint8(np_b)).save('result/LaplaceMask2.bmp')

    #(c) Minimum variance Laplacian: 20
    np_c = getMinVarLaplacian(np_img_padding1, threshold = 20)
    Image.fromarray(np.uint8(np_c)).save('result/LaplaceMinimumVariance.bmp')

    np_img_padding5 = getPadding(getPadding(getPadding(getPadding(np_img_padding1))))
    #(d) Laplace of Gaussian: 3000
    np_d = getLaplacianOfGaussian(np_img_padding5, threshold = 3000)
    Image.fromarray(np.uint8(np_d)).save('result/LaplaceOfGaussian.bmp')

    #(e) Difference of Gaussian: 1
    np_e = getDifferenceOfGaussian(np_img_padding5, threshold = 1)
    Image.fromarray(np.uint8(np_e)).save('result/DifferenceOfGaussian.bmp')

if __name__ == '__main__':
    main()