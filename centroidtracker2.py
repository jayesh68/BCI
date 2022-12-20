import cv2
import numpy as np
import time
import math
import scipy
import scipy.fft
from scipy.spatial import ConvexHull
#print(cv2.__version__)
from pyzbar.pyzbar import decode
from scipy.spatial import distance as dist
from collections import OrderedDict
from collections import defaultdict
from itertools import chain
import scipy.spatial as spatial
import scipy.cluster as cluster
from statistics import mean
import imutils
import collections
from KalmanFilter1 import KalmanFilter1

class Track(object):
    def __init__(self, object, trackIdCount):
        self.object_id = trackIdCount 
        self.KF = KalmanFilter1()  
        self.centroids = np.asarray(object)  

class CentroidTracker():
    def __init__(self):
        self.nextObjectID = 0
        self.objects = []
     
    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects.append(Track(centroid,self.nextObjectID))
        self.nextObjectID += 1
        
    def update(self, centroids):
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(centroids),2),dtype="int")
        # loop over the bounding box rectangle
        D=[]
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
            objectIDs = []
            objectCentroids = []

            for i in range(0,len(self.objects)):
                objectIDs.append(self.objects[i].object_id)
                objectCentroids.append(self.objects[i].centroids)
            
            # print('Input Centroids',inputCentroids)
            # print('Object Centroids',np.array(objectCentroids))
            # objectCentroids = list(self.objects.values())
            #print('object IDs',objectIDs,objectCentroids)
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # print('D',D)
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
            c=0
            for (row, col) in zip(rows, cols):
                self.objects[c].KF.predict()
                if row in usedRows or col in usedCols:
                    # print('used')
                    self.objects[i].KF.correct(np.array([[0], [0]]), 0)
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                #print('ID',row)
                objectID = objectIDs[row]
                #print('Object ID',objectID)
                for i in range(0,len(self.objects)):
                    if self.objects[i].object_id==objectID:
                        # print('Inside correction')
                        self.objects[i].centroids=self.objects[i].KF.correct(inputCentroids[col],1)
                        self.objects[i].KF.lastResult = self.objects[i].centroids
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)
                c+=1
            
        return self.objects,D