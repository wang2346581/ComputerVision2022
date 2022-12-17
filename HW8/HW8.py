import math
import random
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

def opening(np_img, kernel):
    return dilation(erosion(np_img, kernel), kernel)

def closing(np_img, kernel):
    return erosion(dilation(np_img, kernel), kernel)

def getGaussianNoise(img, amplitude):
    resImg = np.copy(img)
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            noiseValue = img[i][j] + amplitude * random.gauss(0, 1)
            noiseValue = 255 if noiseValue > 255 else noiseValue
            resImg[i][j] = noiseValue
    return resImg

def getSaltAndPepperNoise(img, threshold):
    resImg = np.copy(img)
    row, col = img.shape
    for i in range(row):
        for j in range(col):
            randomValue = random.uniform(0, 1)
            if (randomValue <= threshold):   
                resImg[i][j] = 0
            elif (randomValue >= (1-threshold)):
                resImg[i][j] = 255
    return resImg

def getPadding(img):
    row, col = img.shape
    resImg = np.zeros((row + 2, col + 2), dtype=int)
    resRow, resCol = resImg.shape
    resImg[0][0] = img[0][0] # up, left
    resImg[0][resCol - 1] = img[0][col - 1] # up, right
    resImg[resRow - 1][0] = img[row - 1][0] # bottom, left
    resImg[resRow - 1][resCol - 1] = img[row - 1][col - 1] # bottom, right
    #print("[4 corner]\n", resImg, resImg.shape, "\n")
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

def getSNR(noiseImg, oriImg):
    # the snr of lena and median_5x5 is about 15.67369
    row, col = noiseImg.shape
    newNoiseImg = np.copy(noiseImg).astype(np.float)
    newOriImg = np.copy(oriImg).astype(np.float)
    # [0, 255] normalize to [0, 1]
    for i in range(row):
        for j in range(col):
            newOriImg[i][j] = newOriImg[i][j] / 255
            newNoiseImg[i][j] = newNoiseImg[i][j] / 255
    mu_s = 0
    mu_Noise = 0
    for i in range(row):
        for j in range(col):
            mu_s += newOriImg[i][j]
            mu_Noise += newNoiseImg[i][j] - newOriImg[i][j]
    mu_s = mu_s / (row * col)
    mu_Noise = mu_Noise / (row * col)
    var_s = 0
    var_Noise = 0
    for i in range(row):
        for j in range(col):
            var_s += (newOriImg[i][j] - mu_s) ** 2
            var_Noise += (newNoiseImg[i][j] - newOriImg[i][j] - mu_Noise) ** 2
    var_s = var_s / (row * col)
    var_Noise = var_Noise / (row * col)
    SNR = 20 * math.log(math.sqrt(var_s / var_Noise), 10)
    return SNR

def boxFilter(img, filterSize = 3):
    row = img.shape[0]
    col = img.shape[1]
    tmpImg = np.zeros((row, col), dtype=int)
    pitch = filterSize // 2
    kernel = []
    for i in range(-pitch, pitch + 1):
        for j in range(-pitch, pitch + 1):
            kernel.append((i, j))
    kernel_size = len(kernel)
    for i in range(pitch, row - pitch):
        for j in range(pitch, col - pitch):
            for k in kernel:
                rowShift, colShift = k
                tmpImg[i][j] += img[i + rowShift][j + colShift]
            tmpImg[i][j] = tmpImg[i][j] / kernel_size
    resImg = np.zeros((row-2*pitch, col-2*pitch), dtype=int)
    resRow, resCol = resImg.shape
    for i in range(resRow):
        for j in range(resCol):
            resImg[i][j] = tmpImg[i + pitch][j + pitch]
    return resImg

def medianFilter(img, filterSize = 3):
    row = img.shape[0]
    col = img.shape[1]
    tmpImg = np.zeros((row, col), dtype=int)
    pitch = filterSize // 2
    kernel = []
    for i in range(-pitch, pitch + 1):
        for j in range(-pitch, pitch + 1):
            kernel.append((i, j))
    for i in range(pitch, row - pitch):
        for j in range(pitch, col - pitch):
            tmp = list()
            for k in kernel:
                rowShift, colShift = k
                tmp.append(img[i + rowShift][j + colShift])
            tmpImg[i][j] = np.median(tmp)
    resImg = np.zeros((row-2*pitch, col-2*pitch), dtype=int)
    resRow, resCol = resImg.shape
    for i in range(resRow):
        for j in range(resCol):
            resImg[i][j] = tmpImg[i + pitch][j + pitch]
    return resImg

def problemC(np_img, gaussian10_pad1, gaussian10_pad2, gaussian30_pad1, gaussian30_pad2,
    spDot05_pad1, spDot05_pad2, spDot10_pad1, spDot10_pad2):
    # gaussian
    boxFilterGaussian10_3 = boxFilter(gaussian10_pad1, filterSize = 3)
    Image.fromarray(np.uint8(boxFilterGaussian10_3)).save('result/boxFilterGaussian10_3.bmp')
    print("[SNR - boxFilterGaussian10_3]: ", getSNR(boxFilterGaussian10_3, np_img))

    boxFilterGaussian10_5 = boxFilter(gaussian10_pad2, filterSize = 5)
    Image.fromarray(np.uint8(boxFilterGaussian10_5)).save('result/boxFilterGaussian10_5.bmp')
    print("[SNR - boxFilterGaussian10_5]: ", getSNR(boxFilterGaussian10_5, np_img))

    boxFilterGaussian30_3 = boxFilter(gaussian30_pad1, filterSize = 3)
    Image.fromarray(np.uint8(boxFilterGaussian30_3)).save('result/boxFilterGaussian30_3.bmp')
    print("[SNR - boxFilterGaussian30_3]: ", getSNR(boxFilterGaussian30_3, np_img))

    boxFilterGaussian30_5 = boxFilter(gaussian30_pad2, filterSize = 5)
    Image.fromarray(np.uint8(boxFilterGaussian30_5)).save('result/boxFilterGaussian30_5.bmp')
    print("[SNR - boxFilterGaussian30_5]: ", getSNR(boxFilterGaussian30_5, np_img))

    # salt and pepper
    boxFilterSPDot05_3 = boxFilter(spDot05_pad1, filterSize = 3)
    Image.fromarray(np.uint8(boxFilterSPDot05_3)).save('result/boxFilterSPDot05_3.bmp')
    print("[SNR - boxFilterSPDot05_3]: ", getSNR(boxFilterSPDot05_3, np_img))

    boxFilterSPDot05_5 = boxFilter(spDot05_pad2, filterSize = 5)
    Image.fromarray(np.uint8(boxFilterSPDot05_5)).save('result/boxFilterSPDot05_5.bmp')
    print("[SNR - boxFilterSPDot05_5]: ", getSNR(boxFilterSPDot05_5, np_img))

    boxFilterSPDot10_3 = boxFilter(spDot10_pad1, filterSize = 3)
    Image.fromarray(np.uint8(boxFilterSPDot10_3)).save('result/boxFilterSPDot10_3.bmp')
    print("[SNR - boxFilterSPDot10_3]: ", getSNR(boxFilterSPDot10_3, np_img))

    boxFilterSPDot10_5 = boxFilter(spDot10_pad2, filterSize = 5)
    Image.fromarray(np.uint8(boxFilterSPDot10_5)).save('result/boxFilterSPDot10_5.bmp')
    print("[SNR - boxFilterSPDot10_5]: ", getSNR(boxFilterSPDot10_5, np_img))

def problemD(np_img, gaussian10_pad1, gaussian10_pad2, gaussian30_pad1, gaussian30_pad2,
    spDot05_pad1, spDot05_pad2, spDot10_pad1, spDot10_pad2):
    # gaussian
    medianFilterGaussian10_3 = medianFilter(gaussian10_pad1, filterSize = 3)
    Image.fromarray(np.uint8(medianFilterGaussian10_3)).save('result/medianFilterGaussian10_3.bmp')
    print("[SNR - medianFilterGaussian10_3]: ", getSNR(medianFilterGaussian10_3, np_img))

    medianFilterGaussian10_5 = medianFilter(gaussian10_pad2, filterSize = 5)
    Image.fromarray(np.uint8(medianFilterGaussian10_5)).save('result/medianFilterGaussian10_5.bmp')
    print("[SNR - medianFilterGaussian10_5]: ", getSNR(medianFilterGaussian10_5, np_img))

    medianFilterGaussian30_3 = medianFilter(gaussian30_pad1, filterSize = 3)
    Image.fromarray(np.uint8(medianFilterGaussian30_3)).save('result/medianFilterGaussian30_3.bmp')
    print("[SNR - medianFilterGaussian30_3]: ", getSNR(medianFilterGaussian30_3, np_img))

    medianFilterGaussian30_5 = medianFilter(gaussian30_pad2, filterSize = 5)
    Image.fromarray(np.uint8(medianFilterGaussian30_5)).save('result/medianFilterGaussian30_5.bmp')
    print("[SNR - medianFilterGaussian30_5]: ", getSNR(medianFilterGaussian30_5, np_img))

    # salt and pepper
    medianFilterSPDot05_3 = medianFilter(spDot05_pad1, filterSize = 3)
    Image.fromarray(np.uint8(medianFilterSPDot05_3)).save('result/medianFilterSPDot05_3.bmp')
    print("[SNR - medianFilterSPDot05_3]: ", getSNR(medianFilterSPDot05_3, np_img))

    medianFilterSPDot05_5 = medianFilter(spDot05_pad2, filterSize = 5)
    Image.fromarray(np.uint8(medianFilterSPDot05_5)).save('result/medianFilterSPDot05_5.bmp')
    print("[SNR - medianFilterSPDot05_5]: ", getSNR(medianFilterSPDot05_5, np_img))

    medianFilterSPDot10_3 = medianFilter(spDot10_pad1, filterSize = 3)
    Image.fromarray(np.uint8(medianFilterSPDot10_3)).save('result/medianFilterSPDot10_3.bmp')
    print("[SNR - medianFilterSPDot10_3]: ", getSNR(medianFilterSPDot10_3, np_img))

    medianFilterSPDot10_5 = medianFilter(spDot10_pad2, filterSize = 5)
    Image.fromarray(np.uint8(medianFilterSPDot10_5)).save('result/medianFilterSPDot10_5.bmp')
    print("[SNR - medianFilterSPDot10_5]: ", getSNR(medianFilterSPDot10_5, np_img))

def problemE(np_img, gaussian10, gaussian30, saltAndPepperDot05, saltAndPepperDot10):
    kernel = np.array(
        [
            [-2, -1,  0], [-2,  0,  0], [-2,  1,  0],
            [-1, -2,  0], [-1, -1,  0], [-1,  0,  0], [-1,  1,  0], [-1,  2,  0],
            [ 0, -2,  0], [ 0, -1,  0], [ 0,  0,  0], [ 0,  1,  0], [ 0,  2,  0],
            [ 1, -2,  0], [ 1, -1,  0], [ 1,  0,  0], [ 1,  1,  0], [ 1,  2,  0],
            [ 2, -1,  0], [ 2,  0,  0], [ 2,  1,  0]
        ]
    )
    # gaussian
    Opening_Closing_gaussian10 = closing(opening(gaussian10, kernel), kernel)
    Image.fromarray(np.uint8(Opening_Closing_gaussian10)).save('result/Opening_Closing_gaussian10.bmp')
    print("[SNR - Opening_Closing_gaussian10]: ", getSNR(Opening_Closing_gaussian10, np_img))

    Closing_Opening_gaussian10 = opening(closing(gaussian10, kernel), kernel)
    Image.fromarray(np.uint8(Closing_Opening_gaussian10)).save('result/Closing_Opening_gaussian10.bmp')
    print("[SNR - Closing_Opening_gaussian10]: ", getSNR(Closing_Opening_gaussian10, np_img))

    Opening_Closing_gaussian30 = closing(opening(gaussian30, kernel), kernel)
    Image.fromarray(np.uint8(Opening_Closing_gaussian30)).save('result/Opening_Closing_gaussian30.bmp')
    print("[SNR - Opening_Closing_gaussian30]: ", getSNR(Opening_Closing_gaussian30, np_img))

    Closing_Opening_gaussian30 = opening(closing(gaussian30, kernel), kernel)
    Image.fromarray(np.uint8(Closing_Opening_gaussian30)).save('result/Closing_Opening_gaussian30.bmp')
    print("[SNR - Closing_Opening_gaussian30]: ", getSNR(Closing_Opening_gaussian30, np_img))

    # salt and pepper
    Opening_Closing_SPDot05 = closing(opening(saltAndPepperDot05, kernel), kernel)
    Image.fromarray(np.uint8(Opening_Closing_SPDot05)).save('result/Opening_Closing_SPDot05.bmp')
    print("[SNR - Opening_Closing_SPDot05]: ", getSNR(Opening_Closing_SPDot05, np_img))

    Closing_Opening_SPDot05 = opening(closing(saltAndPepperDot05, kernel), kernel)
    Image.fromarray(np.uint8(Closing_Opening_SPDot05)).save('result/Closing_Opening_SPDot05.bmp')
    print("[SNR - Closing_Opening_SPDot05]: ", getSNR(Closing_Opening_SPDot05, np_img))
   
    Opening_Closing_SPDot10 = closing(opening(saltAndPepperDot10, kernel), kernel)
    Image.fromarray(np.uint8(Opening_Closing_SPDot10)).save('result/Opening_Closing_SPDot10.bmp')
    print("[SNR - Opening_Closing_SPDot10]: ", getSNR(Opening_Closing_SPDot10, np_img))

    Closing_Opening_SPDot10 = opening(closing(saltAndPepperDot10, kernel), kernel)
    Image.fromarray(np.uint8(Closing_Opening_SPDot10)).save('result/Closing_Opening_SPDot10.bmp')
    print("[SNR - Closing_Opening_SPDot10]: ", getSNR(Closing_Opening_SPDot10, np_img))

def main():
    img = Image.open('lena.bmp')
    np_img = np.array(img, dtype=np.int)
    print("HW8: ", np_img.shape)

    # (a) Generate noisy images with gaussian noise(amplitude of 10 and 30)
    gaussian10 = getGaussianNoise(np_img, 10)
    Image.fromarray(np.uint8(gaussian10)).save('result/gaussian10.bmp')
    print("[SNR - gaussian10]: ", getSNR(gaussian10, np_img))
    gaussian30 = getGaussianNoise(np_img, 30)
    Image.fromarray(np.uint8(gaussian30)).save('result/gaussian30.bmp')
    print("[SNR - gaussian30]: ", getSNR(gaussian30, np_img))

    # (b) Generate noisy images with salt-and-pepper noise( probability 0.1 and 0.05)
    saltAndPepperDot05 = getSaltAndPepperNoise(np_img, 0.05)
    Image.fromarray(np.uint8(saltAndPepperDot05)).save('result/saltAndPepperDot05.bmp')
    print("[SNR - saltAndPepperDot05]: ", getSNR(saltAndPepperDot05, np_img))
    saltAndPepperDot10 = getSaltAndPepperNoise(np_img, 0.10)
    Image.fromarray(np.uint8(saltAndPepperDot10)).save('result/saltAndPepperDot10.bmp')
    print("[SNR - saltAndPepperDot10]: ", getSNR(saltAndPepperDot10, np_img))

    # tmp: Generate padding
    gaussian10_pad1 = getPadding(gaussian10)
    gaussian30_pad1 = getPadding(gaussian30)
    spDot05_pad1 = getPadding(saltAndPepperDot05)
    spDot10_pad1 = getPadding(saltAndPepperDot10)
    gaussian10_pad2 = getPadding(gaussian10_pad1)
    gaussian30_pad2 = getPadding(gaussian30_pad1)
    spDot05_pad2 = getPadding(spDot05_pad1)
    spDot10_pad2 = getPadding(spDot10_pad1)

    # (c) Use the 3x3, 5x5 box filter on images generated by (a)(b)
    problemC(np_img, gaussian10_pad1, gaussian10_pad2, gaussian30_pad1, gaussian30_pad2,
        spDot05_pad1, spDot05_pad2, spDot10_pad1, spDot10_pad2)

    # (d) Use 3x3, 5x5 median filter on images generated by (a)(b)
    problemD(np_img, gaussian10_pad1, gaussian10_pad2, gaussian30_pad1, gaussian30_pad2,
        spDot05_pad1, spDot05_pad2, spDot10_pad1, spDot10_pad2)

    # (e) Use both opening-then-closing and closing-then opening filter
    # (using the octogonal 3-5-5-5-3 kernel, value = 0) on images generated by (a)(b)
    problemE(np_img, gaussian10, gaussian30, saltAndPepperDot05, saltAndPepperDot10)

if __name__ == '__main__':
    main()

#cv2 padding method
#import cv2
#np_imgP = cv2.copyMakeBorder(np_img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)