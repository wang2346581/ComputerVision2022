import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main():
    img = Image.open('lena.bmp')
    #img.show()
    np_img = np.array(img)
    print("HW3: ", np_img.shape)
    row = np_img.shape[0]
    col = np_img.shape[1]


    # (a) original image and its histogram
    hist_dict = dict()
    hist_list = list()
    np_img_a = copy.deepcopy(np_img)
    for i in range(row):
        for j in range(col):
            if(np_img_a[i][j] not in hist_dict):
                hist_dict[np_img_a[i][j]] = 1
            else:
                hist_dict[np_img_a[i][j]] = hist_dict[np_img_a[i][j]] + 1
            hist_list.append(np_img_a[i][j])
    # plot or save the histogram
    plt.hist(hist_list, bins=256)
    plt.savefig('result/histogram_a.png')


    # (b) image with intensity divided by 3 and its histogram
    hist_dict_b = dict()
    hist_list_b = list()
    np_img_b = copy.deepcopy(np_img)
    np_img_b.astype(int)
    for i in range(row):
        for j in range(col):
            np_img_b[i][j] = np_img_b[i][j] / 3
            if(np_img_b[i][j] not in hist_dict_b):
                hist_dict_b[np_img_b[i][j]] = 1
            else:
                hist_dict_b[np_img_b[i][j]] = hist_dict_b[np_img_b[i][j]] + 1
            hist_list_b.append(np_img_b[i][j])
    # plot or save the histogram
    img_b = Image.fromarray(np_img_b)
    img_b.save("result/image-with-intensity-divide-by-3.bmp")
    plt.clf()
    plt.hist(hist_list_b, bins=256, range=[0, 255])
    plt.savefig('result/histogram_b.png')


    # (c) image after applying histogram equalization to (b) and its histogram
    record_list_c = np.zeros(256, dtype=int)
    prob_list_c = np.zeros(256, dtype=float)
    total_pixel = row * col
    max_luminance = 255
    np_img_c = copy.deepcopy(np_img_b) # applying histogram equalization to (b)
    np_img_c.astype(int)
    for i in range(row):
        for j in range(col):
            record_list_c[np_img_c[i][j]] = record_list_c[np_img_c[i][j]] + 1

    for i in range(record_list_c.shape[0]):
        prob_list_c[i] = record_list_c[i]/(total_pixel)

    # get cdf of the img
    prefix_sum = 0
    cdf_list_c = np.zeros(256, dtype=float)
    for i in range(prob_list_c.shape[0]):
        prefix_sum += prob_list_c[i]
        cdf_list_c[i] = prefix_sum

    for i in range(row):
        for j in range(col):
            np_img_c[i][j] = cdf_list_c[np_img_c[i][j]] * max_luminance

    img_c = Image.fromarray(np_img_c)
    img_c.save("result/image-after-applying-histogram-equalization.bmp")

    hist_dict_c = dict()
    hist_list_c = list()
    for i in range(row):
        for j in range(col):
            if(np_img_c[i][j] not in hist_dict_c):
                hist_dict_c[np_img_c[i][j]] = 1
            else:
                hist_dict_c[np_img_c[i][j]] = hist_dict_c[np_img_c[i][j]] + 1
            hist_list_c.append(np_img_c[i][j])

    plt.clf()
    plt.hist(hist_list_c, bins=256, range=[0, 255])
    plt.savefig('result/histogram_c.png')


if __name__ == '__main__':
    main()