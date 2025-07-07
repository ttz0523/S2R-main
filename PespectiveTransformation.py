import os

import numpy as np
import cv2
from matplotlib import pyplot as plt


def my_pres_trans(img, X, Y, m, n):
    width = 2448
    height = 2448

    x = np.array([1, width, width, 1], dtype=np.float32)
    y = np.array([1, 1, height, height], dtype=np.float32)

    src_pts = np.float32([[X[0], Y[0]], [X[1], Y[1]], [X[2], Y[2]], [X[3], Y[3]]])
    dst_pts = np.float32([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    I = cv2.warpPerspective(img, M, (width, height))

    I = cv2.resize(I, (m, n))


    return I


def my_sort(loc_x, loc_y):
    n = max(loc_x) + min(loc_x)
    m = max(loc_y) + min(loc_y)

    X = np.zeros(4)
    Y = np.zeros(4)

    for i in range(4):
        if loc_x[i] < n / 2 and loc_y[i] < m / 2:
            Y[0] = loc_y[i]
            X[0] = loc_x[i]
        elif loc_x[i] < n / 2 and loc_y[i] > m / 2:
            Y[3] = loc_y[i]
            X[3] = loc_x[i]
        elif loc_x[i] > n / 2 and loc_y[i] > m / 2:
            Y[2] = loc_y[i]
            X[2] = loc_x[i]
        elif loc_x[i] > n / 2 and loc_y[i] < m / 2:
            Y[1] = loc_y[i]
            X[1] = loc_x[i]

    return X, Y


if __name__ == '__main__':
    # foreach or forall
    mode = "forall"
    change_name = True
    n = 512
    m = 512

    folder_path = r'results/SCNG_test/test_en'
    new_folder_path = r'results/SCNG_test/test_pes'
    flag = 0
    num = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.JPG')):

            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)

            if flag == 0:
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.title('Click four points')

                loc_x = []
                loc_y = []

                for p in range(1, 5):
                    point = plt.ginput(1, timeout=0)[0]
                    loc_x.append(point[0])
                    loc_y.append(point[1])

                    plt.plot(point[0], point[1], 'r+')
                    plt.draw()

                plt.close()

                loc_x = np.array(loc_x)
                loc_y = np.array(loc_y)

                loc_x = np.floor(loc_x).astype(int)
                loc_y = np.floor(loc_y).astype(int)

                X, Y = my_sort(loc_x, loc_y)

            if mode == "foreach":
                transformed_img = my_pres_trans(image, X, Y, m, n)

            elif mode == "forall":
                transformed_img = my_pres_trans(image, X, Y, m, n)
                flag = 1

            if change_name is True:
                filename = os.path.join('encoded_{}.png'.format(num))
            cv2.imwrite(os.path.join(new_folder_path, filename), transformed_img)
            num += 1
