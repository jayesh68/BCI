import sys
import numpy as np
import cv2
from homograpy import *
from utilitiesGrid4 import *
from GridLinesStore import *
import math
import scipy
import scipy.fft
from scipy.spatial import ConvexHull
import pickle

def CheckCellContour(Cells, x, y):
    for key, value in Cells.items():
        # print(key,type(value),x,y)
        pt=tuple([int(round(x)), int(round(y))])
        dist = cv2.pointPolygonTest(value, pt, False)
        if dist >= 0:
            print('returning',dist,key,pt)
            return key

def processBlueBoxes (img2, img, img3, ct, Cells):
    centroids_blue = []
    box=[]
    boxes = []
    hulls = []
    approxes=[]
    boxes1 = []
    flag = 0
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    low = np.array([106, 70, 49])
    up = np.array([159, 191, 170])
    thresh = cv2.inRange(hsv, low, up)

    cv2.imshow('Blue Objects', thresh)
    kernel1 = np.ones((11, 11), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel1)
    for i in range (0,1):
        thresh = cv2.dilate(thresh, kernel2)

    cv2.imshow("Thresh", thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print('Contour Length', len(contours))
    cnt=contours[0]
    max_area=cv2.contourArea(cnt)
    print(len(contours))
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            continue

        # cv2.drawContours(img, [c], -1, (255, 125, 155), 2)
        centroids_blue.append((cX, cY))
        # print('Contours',c)
        # rect = cv2.minAreaRect(c)
        # # hull = cv2.convexHull(c)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # hull = np.int0(hull)
        # hulls.append(hull)

        left = tuple(c[c[:, :, 0].argmin()][0])
        right = tuple(c[c[:, :, 0].argmax()][0])
        top = tuple(c[c[:, :, 1].argmin()][0])
        bottom = tuple(c[c[:, :, 1].argmax()][0])
        box.append(left)
        box.append(bottom)
        box.append(right)
        box.append(top)
        boxes.append(box)
        box=[]
        print(box)
        # perimeter = cv2.arcLength(cnt, True)
        # epsilon = 0.01 * cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, epsilon, True)
        # print(approx)
        # approx = np.int0(approx)
        # approxes.append(approx)

        #
        # if cv2.contourArea(c)>max_area:
        #     cnt=c
        #     max_area=cv2.contourArea(c)

        cv2.drawContours(img, [c], 0, (255, 0, 255), 2)
    #     # cv2.putText(img, str(text), (int(cX),int(cY)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(22, 235, 55), 2)
    #     # cv2.putText(img, str(text1), (int(box[0][0]),int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (22, 235, 55), 2)
    #     # cv2.putText(img, str(text2), (int(box[1][0]),int(box[1][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(22, 235, 55), 2)
    #     # cv2.putText(img, str(text3), (int(box[2][0]),int(box[2][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(22, 235, 55), 2)
    #     # cv2.putText(img, str(text4), (int(box[3][0]),int(box[3][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(22, 235, 55), 2)

    objects1, D = ct.update(centroids_blue)
    blue_points = collections.defaultdict(list)
    print('Big D',D)
    i = 0
    print(len(objects1),len(centroids_blue))
    for i in range(0, len(objects1)):
        # print(objectID,tuple(centroid))
        # print('ut grid', objects1[i].centroids)
        centroid = tuple(objects1[i].centroids)
        print(centroid)
        text = "Blue {}".format(objects1[i].object_id)
        # center1=(blue,cX1,cY1)

        # print('cent', text, centroid)
        blue_points[str(objects1[i].object_id)].append(objects1[i].centroids)

        # print(objects1[i].centroids)
        cv2.circle(img, (int(objects1[i].centroids[0]), int(objects1[i].centroids[1])), 3, (255, 0, 255), 2)

        # for box in boxes:
        #     min_x=min(box[0][0],box[1][0],box[2][0],box[3][0])
        #     max_x=max(box[0][0],box[1][0],box[2][0],box[3][0])
        #     min_y=min(box[0][1],box[1][1],box[2][1],box[3][1])
        #     max_y=max(box[0][1],box[1][1],box[2][1],box[3][1])
        #
        #     if centroid[0] > min_x and centroid[1] > min_y and centroid[0] < max_x and centroid[1] < max_y:
        #         for corner in box:
        #             x, y = corner[0], corner[1]
        #             blue_points[str(objects1[i].object_id)].append(tuple(corner))
        #         break

        for box in boxes:
            min_x=min(box[0][0],box[1][0],box[2][0],box[3][0])
            max_x=max(box[0][0],box[1][0],box[2][0],box[3][0])
            min_y=min(box[0][1],box[1][1],box[2][1],box[3][1])
            max_y=max(box[0][1],box[1][1],box[2][1],box[3][1])
            if centroid[0] > min_x and centroid[1] > min_y and centroid[0] < max_x and centroid[1] < max_y:
                for corner in box:
                    print(corner)
                    x, y = corner[0], corner[1]
                    blue_points[str(objects1[i].object_id)].append(tuple(corner))
                break
        # for approx in approxes:
        #     # print(hull)
        #     for corner in approx:
        #         print('corner',corner)
        #         corner=corner[0]
        #         x, y = corner[0], corner[1]
        #         blue_points[str(objects1[i].object_id)].append(tuple(corner))
        #     break

        cv2.putText(img, text, (int(centroid[0]) - 20, int(centroid[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (22, 235, 55), 2)
        cv2.drawContours(img, [c], -1, (255, 125, 55), 2)
        cv2.circle(img, tuple(centroid), 3, (255, 125, 55), -1)

    # cv2.imshow('Centroids', img)

    text_align1 = 0
    c1_count, c2_count, c3_count, c4_count, c5_count, c6_count, c7_count = 0, 0, 0, 0, 0, 0, 0
    r1_count, r2_count, r3_count, r4_count, r5_count, r6_count, r7_count = 0, 0, 0, 0, 0, 0, 0

    for key, values in blue_points.items():
        for row in values:
            x, y = row[0], row[1]
            # print(c1,c2,c3,c4,c5,c6,c7,r1,r2,r3,r4,r5,r6,r7,x,y)
            # print(x,y)
            cell = CheckCellContour(Cells, x, y)

            if cell != None:
                c1_count += 1 if cell[2:4] == "C1" else 0
                c2_count += 1 if cell[2:4] == "C2" else 0
                c3_count += 1 if cell[2:4] == "C3" else 0
                c4_count += 1 if cell[2:4] == "C4" else 0
                c5_count += 1 if cell[2:4] == "C5" else 0
                c6_count += 1 if cell[2:4] == "C6" else 0
                c7_count += 1 if cell[2:4] == "C7" else 0

                r1_count += 1 if cell[0:2] == "R1" else 0
                r2_count += 1 if cell[0:2] == "R2" else 0
                r3_count += 1 if cell[0:2] == "R3" else 0
                r4_count += 1 if cell[0:2] == "R4" else 0
                r5_count += 1 if cell[0:2] == "R5" else 0
                r6_count += 1 if cell[0:2] == "R6" else 0
                r7_count += 1 if cell[0:2] == "R7" else 0

        column=[c1_count,c2_count,c3_count,c4_count,c5_count,c6_count,c7_count]
        row=[r1_count,r2_count,r3_count,r4_count,r5_count,r6_count,r7_count]
        c1_count,c2_count,c3_count,c4_count,c5_count,c6_count,c7_count=0,0,0,0,0,0,0
        r1_count,r2_count,r3_count,r4_count,r5_count,r6_count,r7_count=0,0,0,0,0,0,0

        column_count = len([i for i in column if i > 0])
        row_count = len([i for i in row if i > 0])

        col_num = []
        row_num = []
        text=[]
        text1=[]
        print(key,column_count,row_count,column,row)
        if column_count > 1 and row_count == 1:
            orientation = "Horizontal"
            text1 = "Blue {}".format(int(key)) + " " + "LEA:" + " " + orientation
            for i in range(0, len(row)):
                if row[i] > 0:
                    row_num.append(i + 1)

            for i in range(0, len(column)):
                if column[i] > 0:
                    col_num.append(i + 1)

            converted_list = ["R" + str(element) for element in row_num]
            rows = ",".join(converted_list)
            converted_list = ["C" + str(element) for element in col_num]
            cols = ",".join(converted_list)
            text = "Blue {}".format(int(key)) + " " + "in" + " " + rows + " " + cols

        if row_count > 1 and column_count == 1:
            orientation = "Vertical"
            text1 = "Blue {}".format(int(key)) + " " + "LEA:" + " " + orientation
            for i in range(0, len(row)):
                if row[i] > 0:
                    row_num.append(i + 1)

            for i in range(0, len(column)):
                if column[i] > 0:
                    col_num.append(i + 1)

            converted_list = ["R" + str(element) for element in row_num]
            rows = ",".join(converted_list)
            converted_list = ["C" + str(element) for element in col_num]
            cols = ",".join(converted_list)
            text = "Blue {}".format(int(key)) + " " + "in" + " " + rows + " " + cols

        print(text1)
        # text1, text, row_text, col_text, orientation, object_id = assign_row_col(angle, key, frame_count, flag, row,
        #                                                                          column)

        if text1 != [] and text != [] and rows != [] and cols != []:
            cv2.putText(img3, text1, (329, text_align1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img3, text, (29, text_align1 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # print('x',text_align1)
            text_align1 += 30

def main() :

    ct = CentroidTracker()

    key = ' '
    count=0

    image_orig = cv2.imread('../ObjectsBoard.png')
    cv2.imshow("Original Image", image_orig)

    #Extracting relevant area for processing
    img = image_orig[25 : 449, 400: 886]
    img2 = img.copy()
    cv2.imshow('ROI', img)

    intersection_points=np.loadtxt('intersection_points.txt')
    H = find_homography(intersection_points,[[0, 0], [img.shape[1], 0], [0, img.shape[0]],[img.shape[1],img.shape[0]]])
    img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    img=cv2.flip(img,1)
    img2=img.copy()
    img4 = img.copy()
    cv2.imshow('homography', img)

    img3 = np.zeros([556,812,3],dtype=np.uint8)
    img3.fill(255)

    f=open("../Cell_Labels.obj", "rb")
    Cells = pickle.load(f)
    f.close()

    for key,val in Cells.items():
        print(key,val)
        cv2.drawContours(img4, [val], -1, (0, 255, 0), 3)
    cv2.imshow('Contours', img4)

    processBlueBoxes(img2, img, img3, ct, Cells)
    cv2.imshow('Object Tracker', img)

    cv2.imshow('Positions', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()