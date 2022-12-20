import sys
import numpy as np
import cv2
from homograpy import *
import math
import scipy
import scipy.fft
from scipy.spatial import ConvexHull
import pickle
import pyzed.sl as sl
from centroidtracker2 import CentroidTracker
import collections
import time

def main() :

    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 30

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.STANDARD

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    count=0
    key=""
    # print('outside')
    while key!= "113":
        err = zed.grab(runtime)
        # print('inside')
        if err == sl.ERROR_CODE.SUCCESS:            
            # Retrieve the left image, depth image in the half-resolution
            zed.retrieve_image(image_zed, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)

        count+=1
        print('count',count)
        image_orig = image_zed.get_data()
        cv2.imshow('Image_orig',image_orig)
        cv2.imwrite("BCI/InputFrames/frame%d.png" % count, image_orig)

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    
        time.sleep(5)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()