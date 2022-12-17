import numpy as np
from PIL import Image

def h(b, c , d, e):
    res = 's'
    if (b == c):
        if (d != b or e != b):
            res = 'q'
        elif (d == b and e == b):
            res = 'r'
    return res

def f(a):
    if a == "rrrr":
        return 5
    else:
        count = 0
        for i in a:
            if i == 'q':
                count += 1
        return count

def mark_yokoi(img):
    row = img.shape[0]
    col = img.shape[1]
    output = np.zeros((row, col), dtype=int)
    for i in range(row):
        for j in range(col):
            # x7  x2  x6 
            # x3  x0  x1
            # x8  x4  x5
            # use the shape to check the border of image
            x0 = img[i][j]
            x1 = 0 if j == col - 1 else img[i][j + 1]
            x2 = 0 if i == 0 else img[i - 1][j]
            x3 = 0 if j == 0 else img[i][j - 1]
            x4 = 0 if i == row - 1 else img[i + 1][j]
            x5 = 0 if (i == row - 1 or j == col - 1) else img[i + 1][j + 1]
            x6 = 0 if (i == 0 or j == col - 1) else img[i - 1][j + 1]
            x7 = 0 if (i == 0 or j == 0) else img[i - 1][j - 1]
            x8 = 0 if (i == row - 1 or j == 0) else img[i + 1][j - 1]
            a = "" # tag of the Yokoi connectivity
            if x0:
                a += h(x0, x1, x6, x2) # a1 = h(x0, x1, x6, x2)
                a += h(x0, x2, x7, x3) # a2 = h(x0, x2, x7, x3)
                a += h(x0, x3, x8, x4) # a3 = h(x0, x3, x8, x4)
                a += h(x0, x4, x5, x1) # a4 = h(x0, x4, x5, x1)
                out = f(a)
            else:
                out = 7 # the background id is 7
            output[i][j] = out
    return output

def mark_pair(img, m = 1):
    # [input]: m = 1 (Yokoi edge)
    row = img.shape[0]
    col = img.shape[1]
    pair_list = list()
    h = (lambda a, m: 1 if a==m else 0)
    for i in range(row):
        for j in range(col):
            x0 = img[i][j]
            if x0 != 7: # if current point is not background
                x1 = 0 if j == col - 1 else img[i][j + 1]
                x2 = 0 if i == 0 else img[i - 1][j]
                x3 = 0 if j == 0 else img[i][j - 1]
                x4 = 0 if i == row - 1 else img[i + 1][j]
                sum_h = h(x1, m) + h(x2, m) + h(x3, m) + h(x4, m)
                pair_list.append('p' if (sum_h >= 1 and x0 == m) else 'q')
            else:
                pair_list.append('g')
    np_pair = np.array(pair_list).reshape((row, col))
    return np_pair

def mark_shrink(np_pair):
    row = np_pair.shape[0]
    col = np_pair.shape[1]
    output = np.zeros((row, col), np.int)
    h = (lambda _, c, d, e: 1 if ( (c!='g') and ((d=='g')or(e=='g')) ) else 0)
    f = (lambda a1, a2, a3, a4, x: 'g' if (a1+a2+a3+a4) == 1 else x)
    for i in range(row):
        for j in range(col):
            x0 = np_pair[i][j]
            if x0 == 'p':
                x1 = 'g' if j == col - 1 else np_pair[i][j + 1]
                x2 = 'g' if i == 0 else np_pair[i - 1][j]
                x3 = 'g' if j == 0 else np_pair[i][j - 1]
                x4 = 'g' if i == row - 1 else np_pair[i + 1][j]
                x5 = 'g' if (i == row - 1 or j == col - 1) else np_pair[i + 1][j + 1]
                x6 = 'g' if (i == 0 or j == col - 1) else np_pair[i - 1][j + 1]
                x7 = 'g' if (i == 0 or j == 0) else np_pair[i - 1][j - 1]
                x8 = 'g' if (i == row - 1 or j == 0) else np_pair[i + 1][j - 1]
                a1 = h(x0, x1, x6, x2)
                a2 = h(x0, x2, x7, x3)
                a3 = h(x0, x3, x8, x4)
                a4 = h(x0, x4, x5, x1)
                np_pair[i][j] = f(a1, a2, a3, a4, x0)

    for i in range(row):
        for j in range(col):
            if np_pair[i][j] != 'g':
                output[i][j] = 255
    return output

def main():
    img = Image.open('lena.bmp')
    #img.show()
    np_img = np.array(img)
    row = np_img.shape[0]
    col = np_img.shape[1]

    # step1: a binary image (threshold at 128)
    for i in range(row):
        for j in range(col):
            np_img[i][j] = 255 if(np_img[i][j] >= 128) else 0

    # step2: down-sampling from (512, 512) to (64, 64)
    down_size = 64
    np_thin = np.zeros((down_size, down_size), np.int)
    step_row = row // down_size
    step_col = col // down_size
    for i in range(0, row, step_row):
        for j in range(0, col, step_col):
            new_i = i // step_row
            new_j = j // step_col
            np_thin[new_i][new_j] = np_img[i][j]
    #img_tmp = Image.fromarray(np.uint8(np_thin))
    #img_tmp.save("result/down-sampling.bmp")
    thin_row = np_thin.shape[0]
    thin_col = np_thin.shape[0]
    print("HW7: ", np_thin.shape)

    # step3: Thinning Operator as hw requirement
    iteration = 1
    isChanged = True
    while isChanged:
        print("iteration: ", iteration)
        np_ori = np.copy(np_thin)
        isChanged = False
        np_yokoi = mark_yokoi(np_thin)
        np_pair = mark_pair(np_yokoi)
        np_thin = mark_shrink(np_pair)
        img_tmp = Image.fromarray(np.uint8(np_thin))
        img_tmp.save('result/iteration{}.bmp'.format(iteration))

        # check the output is changed from the origin
        for i in range(thin_row):
            for j in range(thin_col):
                if np_ori[i][j] != np_thin[i][j]:
                    isChanged = True
                    break
        iteration += 1

    # step4: output as a file
    fo = open("result/thin.txt", "w")
    for i in range(thin_row):
        line = ""
        for j in range(thin_col):
            if(np_thin[i][j]):
                line += '*'
            else:
                line += ' '
        line += "\n"
        fo.write(line)

if __name__ == '__main__':
    main()