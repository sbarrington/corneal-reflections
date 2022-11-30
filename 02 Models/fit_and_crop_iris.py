'''
FIT AND CROP IRIS 

Code to fit a circle to the iris of an image and crop out the highlighted region.
'''


# Imports
import cv2
import os
import shutil
import random
import face_alignment
import collections
import sys
import dlib
import cv2

import numpy as np
import matplotlib.pyplot as plt

from ipywidgets import IntProgress
from IPython.display import display
from skimage import io
from PIL import Image
from IPython.display import clear_output
from time import sleep
from skimage import feature
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from skimage import feature, transform
from skimage.color import rgb2gray
from tqdm import tqdm
from imutils import face_utils

def fitCircle(P): 
    n  = len(P) # number of points
    M  = np.vstack((P[:,0],P[:,1],np.ones((4)))).T
    b  = np.array([P[:,0]*P[:,0] + P[:,1]*P[:,1]]).T
    u  = np.linalg.pinv(M)@b # solve Mu=b using least-squares
    xc = u[0]/2 # circle center (xc,yc)
    yc = u[1]/2 # 
    r  = np.sqrt(4*u[2] + u[0]*u[0] + u[1]*u[1])/2 # circle radius (r)
    
    return( round(xc[0]), round(yc[0]), round(r[0]) )

# Crop out eyes and write out to file 
def crop_eye(img, eye, radial_padding=10):
    cx = eye[0]
    cy = eye[1]
    r = eye[2]

    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (cx,cy), r+radial_padding, (255,255,255), -1)

    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask[:,:,0]

    # Take all non-white values and make larger 
    is_ = []
    js_ = []

    for i in range(0, len(result)):
        for j in range(0, len(result[i])):
            if result[i][j][3] > 0: # If non-transaprent
                is_.append(i)
                js_.append(j)
    # Return only crop
    crop_boundary = result[np.min(is_):np.max(is_), np.min(js_):np.max(js_)]
    
    return crop_boundary

def extract_pupils(file, input_path, output_path, both_eyes=False, scale_percent = 50):
    file_name = file.split('.')[0]
    file_extension = '.' + file.split('.')[1]
    img = io.imread(input_path+file)
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    #fa  = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', face_detector='sfd')

    #preds = fa.get_landmarks(img)
    
    preds = detector(img, 1)
    
    gray = img
    
    skipped = []
    skipped_n = []
    
    # If face is found
    if( preds is not None ): 
        # If exactly one face is found
        for (i, rect) in enumerate(preds):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            pred = shape
            
            # fit circles to left/right eye landmarks
            eyeL  = fitCircle( pred[[37,38,40,41],:]) # 37,38,40,41
            irisL = img[ eyeL[1]-eyeL[2]:eyeL[1]+eyeL[2], eyeL[0]-eyeL[2]:eyeL[0]+eyeL[2], : ]
            eyeR  = fitCircle( pred[[43,44,47,46],:]) # 43,44,46,47
            irisR = img[ eyeR[1]-eyeR[2]:eyeR[1]+eyeR[2], eyeR[0]-eyeR[2]:eyeR[0]+eyeR[2], : ]
        
    # If more than one face is found:
    elif len(preds) > 1:
        skipped.append(file_name)
        skipped_n.append(len(preds))
        
        return None
    
    # If no faces found:
    else:
        skipped.append(file_name)
        skipped_n.append('none')

        return None
    
    if both_eyes:
        eyes = [eyeL, eyeR]
        
        suff = '_cropped_l'
        for eye in eyes:
            result = crop_eye(img, eye)
            result_img = Image.fromarray(result)
            result_img.save(output_path + file_name + suff + file_extension)
            # resized for viewing ONLY
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
            resized = Image.fromarray(resized)
            resized.save(output_path + 'resized/' + file_name + suff + file_extension)
        
            suff = '_cropped_r'
    
    else: 
        # Take only left eye
        suff = '_cropped_l'
        result = crop_eye(img, eyeL)
        result_img = Image.fromarray(result)

        # FILTER STEP
        #tbc
        
        result_img.save(output_path + file_name + suff + file_extension)
        
        # resized for viewing ONLY
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)
        resized = Image.fromarray(resized)
        resized.save(output_path + 'resized/' + file_name + suff + file_extension)
        

    return result_img