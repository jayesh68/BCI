import sys
import numpy as np
import cv2
from homograpy import *
from utilitiesGrid4 import *
from GridLinesStore import *
import math
import scipy
from GridLinesStore import *
import scipy.fft
from scipy.spatial import ConvexHull
from PIL import Image
import json
import csv
import pickle

def find_hough_lines1(edges, filtered_lines):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120, None, 25)

    if not lines.any():
        print('No lines were found')
        exit()

    if filter:
        rho_threshold = 40
        theta_threshold = 1.2

        # how many lines are similar to a given one
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDICES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]:  # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[
                        indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now

    if filter:
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

    else:
        filtered_lines = lines

    return filtered_lines

def find_hough_lines(edges, filtered_lines):
    lines = cv2.HoughLines(edges, 1, np.pi / 188, 188)

    if not lines.any():
        print('No lines were found')
        exit()

    if filter:
        print('inside')
        rho_threshold = 60.0
        theta_threshold = 10.0

        # how many lines are similar to a given one
        similar_lines = {i: [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i, theta_i = lines[i][0]
                rho_j, theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDICES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x: len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines) * [True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]:  # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)):  # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]:  # and only if we have not disregarded them already
                    continue

                rho_i, theta_i = lines[indices[i]][0]
                rho_j, theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[
                        indices[j]] = False  # if it is similar and have not been disregarded yet then drop it now

    if filter:
        for i in range(len(lines)):  # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])
    else:
        filtered_lines = lines

    return filtered_lines

def disp_row_col(img,top_corners2,left_corners1,right_corners1,lower_corners2):
    c = 1
    # print(top_corners2)
    for i in range(0, len(top_corners2)):
        if c < 8:
            col_name = 'C%d' % c
            # print(col_name)
            cv2.putText(img, col_name, (int(top_corners2[i][0] + 5),
                                        int(top_corners2[i][1] - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (198, 0, 123), 2, cv2.LINE_AA)
        c += 1

    c = 8
    for i in range(0, len(left_corners1)):
        c -= 1
        row_name = 'R%d' % c
        if row_name=="R0":
            break
        cv2.putText(img, row_name, (int(left_corners1[i][0]),
                                    int(left_corners1[i][1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (198, 0, 123), 2, cv2.LINE_AA)

    # c = 0
    # for i in range(0, len(left_corners1)):
    #     c += 1
    #     row_name = 'R%d' % c
    #     cv2.putText(img, row_name, (int(right_corners1[i][0] + 20),
    #                                 int(right_corners1[i][1] + 20)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (198, 0, 123), 2, cv2.LINE_AA)
    #
    # e = 0
    # # print(len(lower_corners2))
    # for i in range(0, len(lower_corners2)):
    #     e += 1
    #     col_name = 'C%d' % e
    #     # print(col_name)
    #     cv2.putText(img, col_name, (int(lower_corners2[i][0] + 30),
    #                                 int(lower_corners2[i][1] + 18)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (198, 0, 123), 2, cv2.LINE_AA)


def processBlueBoxes(img2, img, ct, c1, c2, c3, c4, c5, c6, c7, r1, r2, r3, r4, r5, r6, r7):
    global count, b1, b2, frame_count, frame_count1
    centroids_blue = []
    boxes = []
    boxes1 = []
    flag = 0
    hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    low = np.array([101, 147, 93])
    up = np.array([121, 247, 177])
    thresh = cv2.inRange(hsv, low, up)

    # Creating kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    # Using cv2.erode() method
    kernel1 = np.ones((5, 5), np.uint8)
    for i in range(0, 3):
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # thresh= cv2.dilate(thresh, kernel1)
        # thresh= cv2.erode(thresh, kernel1)

    # cv2.imshow("Thresh",thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print('Contour Length', len(contours))
    contour_pca = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
            continue

        cv2.drawContours(img2, [c], -1, (255, 125, 155), 2)
        centroids_blue.append((cX, cY))
        print(cX, cY)
        rect = cv2.minAreaRect(c)
        # print('Rect',rect)
        box = cv2.boxPoints(rect)
        # print('Box',box)
        box = np.int0(box)
        boxes.append(box)
        boxes1.append(box)
        text = "{},{}".format(cX, cY)
        text1 = "{},{}".format(box[0][0], box[0][1])
        text2 = "{},{}".format(box[1][0], box[1][1])
        text3 = "{},{}".format(box[2][0], box[2][1])
        text4 = "{},{}".format(box[3][0], box[3][1])

        cv2.drawContours(img, [box], 0, (255, 0, 255), 2)

    objects1, D = ct.update(centroids_blue)
    blue_points = collections.defaultdict(list)

    i = 0

    for i in range(0, len(objects1)):
        # print(objectID,tuple(centroid))
        print('ut grid', objects1[i].centroids)
        centroid = tuple(objects1[i].centroids)
        text = "Blue {}".format(objects1[i].object_id)
        # center1=(blue,cX1,cY1)

        print('cent', text, centroid)
        blue_points[str(objects1[i].object_id)].append(objects1[i].centroids)

        print(objects1[i].centroids)
        cv2.circle(img, (int(objects1[i].centroids[0]), int(objects1[i].centroids[1])), 3, (255, 0, 255), 2)

        # count=0
        for box in boxes:
            # print('box',box,centroid[0],centroid[1],box[0][0],box[0][1],box[2][0],box[2][1])
            if centroid[0] > box[0][0] and centroid[1] > box[0][1] and centroid[0] < box[2][0] and centroid[1] < box[2][1]:
                # print('inside')
                for corner in box:
                    x, y = corner[0], corner[1]
                    blue_points[str(objects1[i].object_id)].append(tuple(corner))
                break
                # count+=1

        cv2.putText(img, text, (int(centroid[0]) - 20, int(centroid[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (22, 235, 55), 2)
        # print('object',objectID,centroid)
        # cv2.drawContours(img, [c], -1, (255, 125, 55), 2)
        # cv2.circle(img, tuple(centroid), 3, (255, 125, 55), -1)

    # cv2.imshow('Detected blue',img2)
    angle = []
    for c in contours:
        if len(c) > 1:
            angle.append(getOrientation(c, img))
        else:
            angle.append(0)


def main():
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 30

    image_orig = cv2.imread('NewRes.png')
    lines=[]
    lines1=[]
    top_corners2, top_last = [], []
    left_corners1, left_last = [], []
    right_corners1, right_last = [], []
    lower_corners2, bottom_last = [], []
    ct = CentroidTracker()

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    
    # Extracting relevant area for processing
    img = image_orig[25:449, 400: 886]
    # img2 = img.copy()
    cv2.imshow('Original',image_orig)
    cv2.imshow('ROI', img)
    edges = cv2.Canny(img, 50, 200)
    cv2.imshow('Edges', edges)

    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = find_hough_lines(edges,lines)

    if lines is not None:
            print(len(lines))
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

    cv2.imshow('Hough Lines',cdst)

    lines = np.reshape(lines, (-1, 2))
    h_lines, v_lines = h_v_lines(lines)
    intersection_points = line_intersections(h_lines, v_lines)
    # print(intersection_points)
    # print(img.shape[0],img.shape[1])
    H = find_homography(intersection_points,  [[0, 0], [img.shape[1], 0], [0, img.shape[0]], [img.shape[1],img.shape[0]]])
    img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
    img2=img.copy()
    img3=img.copy()
    img4 = img.copy()
    cv2.imshow('homo.png', img)
    np.savetxt('intersection_points.txt', intersection_points)
    # img = cv2.flip(img, 1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(img_gray,(7, 7),cv2.BORDER_DEFAULT)
    img_binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 23, 19)
    cv2.imshow('Binary',img_binary)

    cnts, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy=hierarchy[0]
    print(len(cnts))
    c=0
    Cell_Contour=[]
    Bad_Cell_Contour=[]
    for j, cnt in zip(hierarchy, cnts):
        # print('j', len(j))
        current_cnt=cnt
        # hierarchy=hierarchy[0]
        if j[2] <= 0:
            c += 1
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawContours(img, [current_cnt], -1, (0, 255, 0), 3)
            cv2.putText(img, str(c), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            if c>=5 and c<=49:
                Cell_Contour.append(current_cnt)
            else:
                Bad_Cell_Contour.append(current_cnt)

    Cell_Contour.insert(0,Bad_Cell_Contour[1])
    Cell_Contour.insert(0,Bad_Cell_Contour[0])
    Cell_Contour.insert(0,Bad_Cell_Contour[3])
    Cell_Contour.insert(0,Bad_Cell_Contour[2])

    c=0
    for cnt in Cell_Contour:
        c+=1
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.drawContours(img2, [cnt], -1, (0, 255, 0), 3)
        cv2.putText(img2, str(c), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Stored Contours',img2)
    cv2.drawContours(img, cnts, -1, (0, 255, 0), 3)
    cv2.imshow('Squares', img)

    Cell_Labels={}
    row=7
    column=7
    # print('Final Contours',len(Cell_Contour))
    c = 0
    for cnt in Cell_Contour:
        c+=1
        # print('Count',c,row,column)
        if column>0:
            Cell_Labels["R"+ str(row)+ "C" + str(column)]=cnt
            # print("R"+ str(row)+ "C" + str(column))
            column-=1
            # continue
        else:
            row-=1
            column=7
            # print("R" + str(row) + "C" + str(column))
            Cell_Labels["R" + str(row) + "C" + str(column)] = cnt
            column-=1

    c=0
    w = csv.writer(open("output.csv", "w"))
    for key,value in Cell_Labels.items():
        c+=1
        w.writerow([key, value])
        # print('Value', value)
        cv2.drawContours(img3, [value], -1, (0, 255, 0), 3)
        M = cv2.moments(value)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(img3, str(c), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Dictionary image',img3)

    f = open("Cell_Labels.obj", 'wb')
    pickle.dump(Cell_Labels, f)
    f.close()

    f=open("Cell_Labels.obj","rb")
    Cells = pickle.load(f)
    f.close()

    print(Cells)
    # with open('output.csv') as f:
    #     Cells=csv.reader(f)
    #
    #     for key,val in Cells:
    #         # print(key, val)
    #         cv2.drawContours(img4, [val], -1, (0, 255, 0), 3)

    for key,val in Cells.items():
        print(key,val)
        cv2.drawContours(img4, [val], -1, (0, 255, 0), 3)

    cv2.imshow('Loaded Dictionary image', img4)

    # json1 = json.dumps(Cell_Labels)
    # f = open("squares.json", "w")
    # f.write(json1)
    # f.close()

    # Now that contour are obtained. All the points representing a contour needs to be stored.
    # print(len(cnts), hierarchy)

    # morph_kernel = np.ones((15, 15), np.uint8)
    # output = cv2.morphologyEX(img_binary_inverted, cv2.MORPH_CLOSE, morph_kernel)
    #
    # cv2.imshow('Morphology', output)

    # img = img[25:412, 25: 466]
    # v = np.median(img)
    # sigma = 0.33
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    # edges1 = cv2.Canny(img, 50, 200,None,3)
    #
    # lines1 = find_hough_lines1(edges1, lines1)
    # # # lines1_reshape = np.reshape(lines1.shape[0],-1)
    #
    # # lines1 = cv2.HoughLinesP(edges, 1, np.pi / 180, 150, None, 20, 10)
    # # np.savetxt('lines1.txt', lines1_reshape)
    # rho_theta=[]
    # if lines1 is not None:
    #     print('not none')
    #     for i in range(0, len(lines1)):
    #         rho, theta = lines1[i][0][0], lines1[i][0][1]
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))
    #         print(x1,y1,' ',x2,y2)
    #         rho_theta.append((rho,theta))
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #
    # cv2.imshow('Grids', img)
    #
    # np.savetxt('lines1.txt', rho_theta)
    # lines1 = np.reshape(lines1, (-1, 2))
    # h_lines, v_lines = h_v_lines(lines1)
    # intersection_points = line_intersections(h_lines, v_lines)
    #
    # intersection_points = intersection_points[intersection_points[:, 0].argsort()]
    # print(len(intersection_points))
    #
    # top_corners2, top_last = get_top_corner_points(intersection_points)
    #
    # intersection_points1 = intersection_points[intersection_points[:, 1].argsort()]
    # left_corners1, left_last = get_left_corner_points(intersection_points1)
    #
    # intersection_points1 = intersection_points[intersection_points[:, 1].argsort()]
    # right_corners1, right_last = get_right_corner_points(intersection_points1)
    #
    # intersection_points = intersection_points[intersection_points[:, 0].argsort()]
    # print(intersection_points)
    #
    # lower_corners2, bottom_last = get_bottom_corner_points(intersection_points)
    # print('top', len(top_corners2), 'left', len(left_corners1), 'right', len(right_corners1), 'lower', len(lower_corners2))
    #
    # disp_row_col(img,top_corners2,left_corners1,right_corners1,lower_corners2)
    #
    # np.savetxt('top_corners2.txt', top_corners2)
    # np.savetxt('left_corners1.txt', left_corners1)
    # np.savetxt('right_corners1.txt', right_corners1)
    # np.savetxt('lower_corners2.txt', lower_corners2)
    #
    # cv2.imshow('Grids Label', img)
    # c1, c2, c3, c4, c5, c6, c7, r1, r2, r3, r4, r5, r6, r7 = line_equation(top_corners2, left_corners1, right_corners1,
    #                                                                        lower_corners2)

    # processBlueBoxes(img2, img, ct, c1, c2, c3, c4, c5, c6, c7, r1, r2, r3, r4, r5, r6, r7)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()