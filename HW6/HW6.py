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

def main():
    img = Image.open('lena.bmp')
    #img.show()
    np_img = np.array(img)
    print("HW6: ", np_img.shape)
    row = np_img.shape[0]
    col = np_img.shape[1]

    # step1: a binary image (threshold at 128)
    for i in range(row):
        for j in range(col):
            np_img[i][j] = 255 if(np_img[i][j] >= 128) else 0

    # step2: down-sampling from (512, 512) to (64, 64)
    down_size = 64
    new_img = np.zeros((down_size, down_size), np.int)
    step_row = row // down_size
    step_col = col // down_size
    for i in range(0, row, step_row):
        for j in range(0, col, step_col):
            new_i = i // step_row
            new_j = j // step_col
            new_img[new_i][new_j] = np_img[i][j]
    #img_tmp = Image.fromarray(np.uint8(new_img))
    #img_tmp.save("result/down-sampling.bmp")

    # step3: calculating the Yokoi connectivity number by 4-connected component
    output = np.zeros((down_size, down_size), dtype=int)
    out_row = output.shape[0]
    out_col = output.shape[1]

    for i in range(out_row):
        for j in range(out_col):
            # x7  x2  x6 
            # x3  x0  x1
            # x8  x4  x5
            # use the shape to check the border of image
            x0 = new_img[i][j]
            x1 = 0 if j == out_col - 1 else new_img[i][j + 1]
            x2 = 0 if i == 0 else new_img[i - 1][j]
            x3 = 0 if j == 0 else new_img[i][j - 1]
            x4 = 0 if i == out_row - 1 else new_img[i + 1][j]
            x5 = 0 if (i == out_row - 1 or j == out_col - 1) else new_img[i + 1][j + 1]
            x6 = 0 if (i == 0 or j == out_col - 1) else new_img[i - 1][j + 1]
            x7 = 0 if (i == 0 or j == 0) else new_img[i - 1][j - 1]
            x8 = 0 if (i == out_row - 1 or j == 0) else new_img[i + 1][j - 1]
            a = "" # tag of the Yokoi connectivity
            if x0:
                a += h(x0, x1, x6, x2) # a1 = h(x0, x1, x6, x2)
                a += h(x0, x2, x7, x3) # a2 = h(x0, x2, x7, x3)
                a += h(x0, x3, x8, x4) # a3 = h(x0, x3, x8, x4)
                a += h(x0, x4, x5, x1) # a4 = h(x0, x4, x5, x1)
                out = f(a)
            else:
                continue
            output[i][j] = out

    # step4: output as a file
    fo = open("result/yokoi-connectivity.txt", "w")
    for i in range(down_size):
        line = ""
        for j in range(down_size):
            if(output[i][j]):
                line += str(output[i][j])
            else:
                line += ' '
        line += "\n"
        fo.write(line)

if __name__ == '__main__':
    main()