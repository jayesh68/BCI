import cv2
import numpy as np
import time
import math
import scipy
import scipy.fft
from scipy.spatial import ConvexHull
#print(cv2.__version__)
from scipy.spatial import distance as dist
from collections import OrderedDict
from collections import defaultdict
from itertools import chain
import scipy.spatial as spatial
import scipy.cluster as cluster
from statistics import mean
import imutils
import collections

class CentroidTracker():
    def __init__(self):
        self.nextObjectID = 0
        self.objects = OrderedDict()
     
    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.nextObjectID += 1
        
    def update(self, centroids):
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(centroids),2),dtype="int")
        # loop over the bounding box rectangle

        if len(centroids)==5:
            print('hi5')
        #print(len(centroids))
        for (i,centroid) in enumerate(centroids):
        # use the bounding box coordinates to derive the centroid
            #print('centroid',i,centroid)
            inputCentroids[i] = centroid
        
        #print('object count',len(self.objects))
        if len(self.objects)==0:
            #print('registering')
            for i in range(0,len(inputCentroids)):
                self.register(inputCentroids[i])
        else:       
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            #print('object IDs',objectIDs,objectCentroids)
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            #print(D)
            #print('Minimum',D.min(axis=1))
            rows = D.min(axis=1).argsort()
            #print('rows',rows)
            #print('Minimum',D.argmin(axis=1))
            cols = D.argmin(axis=1)[rows]
            cols1=D.argmin(axis=1)
            cols2=cols1[rows]
            #print('rows',rows,'cols',cols)   
            #print('cols',cols1,cols2)
            usedRows = set()
            usedCols = set()
        
            #print(list(zip(rows, cols)))
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    print('used')
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                #print('ID',row)
                objectID = objectIDs[row]
                #print('Object ID',objectID)
                self.objects[objectID] = inputCentroids[col]
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
                
        return self.objects