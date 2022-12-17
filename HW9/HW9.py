import math
import numpy as np
from PIL import Image

def getRobertPadding(img):
    row, col = img.shape
    resImg = np.zeros((row + 1, col + 1), dtype=int)
    resRow, resCol = resImg.shape
    resImg[resRow - 1][resCol - 1] = img[row - 1][col - 1] # bottom, right

    for j in range(resCol - 1): # bottom
        resImg[resRow - 1][j] = img[row - 1][j]

    for i in range(resRow - 1): # right
        resImg[i][resCol - 1] = img[i][col - 1]

    for i in range(resRow - 1): # Remain Pixel
        for j in range(resCol - 1):
            resImg[i][j] = img[i][j]
    return resImg

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

def robertsOperator(img, threshold = 12):
    row = img.shape[0]
    col = img.shape[1]
    resImg = np.zeros((row - 1, col - 1), dtype=int)
    resRow, resCol = resImg.shape
    # kernel: x, y, value
    r1List = np.array([[0, 0, -1], [1, 1, 1]])
    r2List = np.array([[0, 1, -1], [1, 0, 1]])
    for i in range(resRow):
        for j in range(resCol):
            r1Sum = 0
            r2Sum = 0
            for r1 in r1List:
                r1Sum += img[i + r1[0]][j + r1[1]] * r1[2]
            for r2 in r2List:
                r2Sum += img[i + r2[0]][j + r2[1]] * r2[2]
            gradientMagnitude = math.sqrt((r1Sum ** 2) + (r2Sum ** 2))
            resImg[i][j] = 0 if gradientMagnitude >= threshold else 255
    return resImg

def generalOperator(img, threshold, list1, list2):
    row, col = img.shape
    resImg = np.zeros((row - 2, col - 2), dtype=int)
    resRow, resCol = resImg.shape
    for i in range(resRow):
        for j in range(resCol):
            sum1 = 0
            sum2 = 0
            for ele in list1:
                sum1 += img[i + 1 + int(ele[0])][j + 1 + int(ele[1])] * ele[2]
            for ele in list2:
                sum2 += img[i + 1 + int(ele[0])][j + 1 + int(ele[1])] * ele[2]
            gradientMagnitude = math.sqrt((sum1 ** 2) + (sum2 ** 2))
            resImg[i][j] = 0 if gradientMagnitude >= threshold else 255
    return resImg

def prewittOperator(img, threshold = 24):
    # kernel: x, y, value
    p1List = np.array([
        [-1, -1, -1], [-1, 0, -1], [-1, 1, -1],
        [1, -1, 1], [1, 0, 1], [1, 1, 1],
    ])
    p2List = np.array([
        [-1, -1, -1], [-1, 1, 1],
        [0, -1, -1], [0, 1, 1],
        [1, -1, -1], [1, 1, 1],
    ])
    return generalOperator(img, threshold, p1List, p2List)

def sobelOperator(img, threshold = 38):
    # kernel: x, y, value
    s1List = np.array([
        [-1, -1, -1], [-1, 0, -2], [-1, 1, -1],
        [1, -1, 1], [1, 0, 2], [1, 1, 1],
    ])
    s2List = np.array([
        [-1, -1, -1], [-1, 1, 1],
        [0, -1, -2], [0, 1, 2],
        [1, -1, -1], [1, 1, 1],
    ])
    return generalOperator(img, threshold, s1List, s2List)

def freiAndChenOperator(img, threshold = 30):
    # kernel: x, y, value
    sqrt2 = math.sqrt(2)
    f1List = np.array([
        [-1, -1, -1], [-1, 0, -sqrt2], [-1, 1, -1],
        [1, -1, 1], [1, 0, sqrt2], [1, 1, 1],
    ])
    f2List = np.array([
        [-1, -1, -1], [-1, 1, 1],
        [0, -1, -sqrt2], [0, 1, sqrt2],
        [1, -1, -1], [1, 1, 1],
    ])
    return generalOperator(img, threshold, f1List, f2List)

def kirschCompassOperator(img, threshold = 135):
    # kernel: x, y, value
    kList = np.array(
        [
            # k0
            [
                [-1, -1, -3], [-1, 0, -3], [-1, 1, 5],
                [0, -1, -3], [0, 1, 5],
                [1, -1, -3], [1, 0, -3], [1, 1, 5],
            ],
            # k1
            [
                [-1, -1, -3], [-1, 0, 5], [-1, 1, 5],
                [0, -1, -3], [0, 1, 5],
                [1, -1, -3], [1, 0, -3], [1, 1, -3],
            ],
            # k2
            [
                [-1, -1, 5], [-1, 0, 5], [-1, 1, 5],
                [0, -1, -3], [0, 1, -3],
                [1, -1, -3], [1, 0, -3], [1, 1, -3],
            ],
            # k3
            [
                [-1, -1, 5], [-1, 0, 5], [-1, 1, -3],
                [0, -1, 5], [0, 1, -3],
                [1, -1, -3], [1, 0, -3], [1, 1, -3],
            ],
            # k4
            [
                [-1, -1, 5], [-1, 0, -3], [-1, 1, -3],
                [0, -1, 5], [0, 1, -3],
                [1, -1, 5], [1, 0, -3], [1, 1, -3],
            ],
            # k5
            [
                [-1, -1, -3], [-1, 0, -3], [-1, 1, -3],
                [0, -1, 5], [0, 1, -3],
                [1, -1, 5], [1, 0, 5], [1, 1, -3],
            ],
            # k6
            [
                [-1, -1, -3], [-1, 0, -3], [-1, 1, -3],
                [0, -1, -3], [0, 1, -3],
                [1, -1, 5], [1, 0, 5], [1, 1, 5],
            ],
            # k7
            [
                [-1, -1, -3], [-1, 0, -3], [-1, 1, -3],
                [0, -1, -3], [0, 1, 5],
                [1, -1, -3], [1, 0, 5], [1, 1, 5],
            ],
        ]
    )
    row, col = img.shape
    resImg = np.zeros((row - 2, col - 2), dtype=int)
    resRow, resCol = resImg.shape
    for i in range(resRow):
        for j in range(resCol):
            sumList = np.zeros(8, np.float)
            for idx, k in enumerate(kList):
                for ele in k:
                    sumList[idx] += img[i + 1 + int(ele[0])][j + 1 + int(ele[1])] * ele[2]
            gradientMagnitude = np.max(sumList)
            resImg[i][j] = 0 if gradientMagnitude >= threshold else 255
    return resImg

def RobinsonCompassOperator(img, threshold = 43):
    rList = np.array(
        [
            # k0
            [
                [-1, -1, -1], [-1, 1, 1],
                [0, -1, -2], [0, 1, 2],
                [1, -1, -1], [1, 1, 1],
            ],
            # k1
            [
                [-1, 0, 1], [-1, 1, 2],
                [0, -1, -1], [0, 1, 1],
                [1, -1, -2], [1, 0, -1],
            ],
            # k2
            [
                [-1, -1, 1], [-1, 0, 2], [-1, 1, 1],
                [1, -1, -1], [1, 0, -2], [1, 1, -1],
            ],
            # k3
            [
                [-1, -1, 2], [-1, 0, 1],
                [0, -1, 1], [0, 1, -1],
                [1, 0, -1], [1, 1, -2],
            ]
        ]
    )
    row, col = img.shape
    resImg = np.zeros((row - 2, col - 2), dtype=int)
    resRow, resCol = resImg.shape
    for i in range(resRow):
        for j in range(resCol):
            sumList = np.zeros(8, np.float)
            for idx, k in enumerate(rList):
                for ele in k:
                    sumList[idx] += img[i + 1 + int(ele[0])][j + 1 + int(ele[1])] * ele[2]
            for idx in range(4, 8):
                sumList[idx] = -sumList[idx - 4]
            gradientMagnitude = np.max(sumList)
            resImg[i][j] = 0 if gradientMagnitude >= threshold else 255
    return resImg

def NevatiaBabuOperator(img, threshold = 12500):
    nBList = np.array(
        [
            # 0
            [
                [-2, -2,  100], [-2, -1,  100], [-2, 0,  100], [-2, 1,  100], [-2, 2,  100],
                [-1, -2,  100], [-1, -1,  100], [-1, 0,  100], [-1, 1,  100], [-1, 2,  100],
                [ 0, -2,    0], [ 0, -1,    0], [ 0, 0,    0], [ 0, 1,    0], [ 0, 2,    0],
                [ 1, -2, -100], [ 1, -1, -100], [ 1, 0, -100], [ 1, 1, -100], [ 1, 2, -100],
                [ 2, -2, -100], [ 2, -1, -100], [ 2, 0, -100], [ 2, 1, -100], [ 2, 2, -100],
            ],
            # 30
            [
                [-2, -2,  100], [-2, -1,  100], [-2, 0,  100], [-2, 1,  100], [-2, 2,  100],
                [-1, -2,  100], [-1, -1,  100], [-1, 0,  100], [-1, 1,   78], [-1, 2,  -32],
                [ 0, -2,  100], [ 0, -1,   92], [ 0, 0,    0], [ 0, 1,  -92], [ 0, 2, -100],
                [ 1, -2,   32], [ 1, -1,  -78], [ 1, 0, -100], [ 1, 1, -100], [ 1, 2, -100],
                [ 2, -2, -100], [ 2, -1, -100], [ 2, 0, -100], [ 2, 1, -100], [ 2, 2, -100],
            ],
            # 60
            [
                [-2, -2,  100], [-2, -1,  100], [-2, 0,  100], [-2, 1,   32], [-2, 2, -100],
                [-1, -2,  100], [-1, -1,  100], [-1, 0,   92], [-1, 1,  -78], [-1, 2, -100],
                [ 0, -2,  100], [ 0, -1,  100], [ 0, 0,    0], [ 0, 1, -100], [ 0, 2, -100],
                [ 1, -2,  100], [ 1, -1,   78], [ 1, 0,  -92], [ 1, 1, -100], [ 1, 2, -100],
                [ 2, -2,  100], [ 2, -1,  -32], [ 2, 0, -100], [ 2, 1, -100], [ 2, 2, -100],
            ],
            # -90
            [
                [-2, -2, -100], [-2, -1, -100], [-2, 0,    0], [-2, 1,  100], [-2, 2,  100],
                [-1, -2, -100], [-1, -1, -100], [-1, 0,    0], [-1, 1,  100], [-1, 2,  100],
                [ 0, -2, -100], [ 0, -1, -100], [ 0, 0,    0], [ 0, 1,  100], [ 0, 2,  100],
                [ 1, -2, -100], [ 1, -1, -100], [ 1, 0,    0], [ 1, 1,  100], [ 1, 2,  100],
                [ 2, -2, -100], [ 2, -1, -100], [ 2, 0,    0], [ 2, 1,  100], [ 2, 2,  100],
            ],
            # -60
            [
                [-2, -2, -100], [-2, -1,   32], [-2, 0,  100], [-2, 1,  100], [-2, 2,  100],
                [-1, -2, -100], [-1, -1,  -78], [-1, 0,   92], [-1, 1,  100], [-1, 2,  100],
                [ 0, -2, -100], [ 0, -1, -100], [ 0, 0,    0], [ 0, 1,  100], [ 0, 2,  100],
                [ 1, -2, -100], [ 1, -1, -100], [ 1, 0,  -92], [ 1, 1,   78], [ 1, 2,  100],
                [ 2, -2, -100], [ 2, -1, -100], [ 2, 0, -100], [ 2, 1,  -32], [ 2, 2,  100],
            ],
            # -30
            [
                [-2, -2,  100], [-2, -1,  100], [-2, 0,  100], [-2, 1,  100], [-2, 2,  100],
                [-1, -2,  -32], [-1, -1,   78], [-1, 0,  100], [-1, 1,  100], [-1, 2,  100],
                [ 0, -2, -100], [ 0, -1,  -92], [ 0, 0,    0], [ 0, 1,   92], [ 0, 2,  100],
                [ 1, -2, -100], [ 1, -1, -100], [ 1, 0, -100], [ 1, 1,  -78], [ 1, 2,   32],
                [ 2, -2, -100], [ 2, -1, -100], [ 2, 0, -100], [ 2, 1, -100], [ 2, 2, -100],
            ]
        ]
    )
    row, col = img.shape
    resImg = np.zeros((row - 4, col - 4), dtype=int)
    resRow, resCol = resImg.shape
    for i in range(resRow):
        for j in range(resCol):
            sumList = np.zeros(len(nBList), np.float)
            for idx, k in enumerate(nBList):
                for ele in k:
                    sumList[idx] += img[i + 2 + int(ele[0])][j + 2 + int(ele[1])] * ele[2]
            gradientMagnitude = np.max(sumList)
            resImg[i][j] = 0 if gradientMagnitude >= threshold else 255
    return resImg

def main():
    img = Image.open('lena.bmp')
    np_img = np.array(img, dtype=np.int)
    print("HW9: ", np_img.shape)
    np_padding_img = getPadding(np_img)
    np_padding_img2 = getPadding(np_padding_img)

    # (a) Robert's Operator: 12
    rP_img = getRobertPadding(np_img)
    robertImg = robertsOperator(rP_img, 12)
    Image.fromarray(np.uint8(robertImg)).save('result/Robert.bmp')

    # (b) Prewitt's Edge Detector: 24
    prewittImg = prewittOperator(np_padding_img, 24)
    Image.fromarray(np.uint8(prewittImg)).save('result/Prewitt.bmp')

    # (c) Sobel's Edge Detector: 38
    sobelImg = sobelOperator(np_padding_img, 38)
    Image.fromarray(np.uint8(sobelImg)).save('result/Sobel.bmp')

    #(d) Frei and Chen's Gradient Operator: 30
    freiAndChenImg = freiAndChenOperator(np_padding_img, 30)
    Image.fromarray(np.uint8(freiAndChenImg)).save('result/Frei_And_Chen.bmp')

    # (e) Kirsch's Compass Operator: 135
    kirschCompassImg= kirschCompassOperator(np_padding_img, threshold = 135)
    Image.fromarray(np.uint8(kirschCompassImg)).save('result/KirschCompass.bmp')

    # (f) Robinson's Compass Operator: 43
    robinsonCompassImg= RobinsonCompassOperator(np_padding_img, threshold = 43)
    Image.fromarray(np.uint8(robinsonCompassImg)).save('result/RobinsonCompass.bmp')

    #(g) Nevatia-Babu 5x5 Operator: 12500
    nevatiaBabuImg = NevatiaBabuOperator(np_padding_img2, threshold = 12500)
    Image.fromarray(np.uint8(nevatiaBabuImg)).save('result/NevatiaBabu.bmp')

if __name__ == '__main__':
    main()