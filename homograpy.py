# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:11:11 2022

@author: jayes
"""
import numpy as np
import cv2

#Technique to increase the luminance in an image to improve detection performance
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Function to compute the homography from the corner points
def find_homography(img1, img2):
    ind = 0
    A_matrix = np.empty((8, 9))

    for pixel in range(0, len(img1)):
        x_1 = img1[pixel][0]
        y_1 = img1[pixel][1]

        x_2 = img2[pixel][0]
        y_2 = img2[pixel][1]

        A_matrix[ind] = np.array([x_1, y_1, 1, 0, 0, 0, -x_2 * x_1, -x_2 * y_1, -x_2])
        A_matrix[ind + 1] = np.array([0, 0, 0, x_1, y_1, 1, -y_2 * x_1, -y_2 * y_1, -y_2])

        ind = ind + 2

    U, s, V = np.linalg.svd(A_matrix, full_matrices=True)
    V = (V) / (V[8][8])
    H = V[8, :].reshape(3, 3)
    return H