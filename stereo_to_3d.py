#####################################################################

# Example : project SGBM disparity to 3D points for am example pair
# of rectified stereo images from a  directory structure
# of left-images / right-images with filesname DATE_TIME_STAMP_{L|R}.png

# basic illustrative python script for use with provided stereo datasets

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2017 Deparment of Computer Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import os
import numpy as np
import random
import csv

master_path_to_dataset = "TTBB-durham-02-10-17-sub5" # ** need to edit this **
directory_to_cycle_left = "left-images"     # edit this if needed
directory_to_cycle_right = "right-images"   # edit this if needed

#####################################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_centre_w = 474.5;

principle_point = (474.5, 262.0) #Taken from mono_stream.py

camera_matrix = np.array([[camera_focal_length_px, 0, principle_point[0]],
                         [0, camera_focal_length_px, principle_point[1]],
                         [0, 0, 1]])

#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, rgb=[]):

    points = [];

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then get reasonable scaling in X and Y output

    Zmax = ((f * B) / 2);

    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = (f * B) / disparity[y,x];

                X = ((x - image_centre_w) * Zmax) / f;
                Y = ((y - image_centre_h) * Zmax) / f;

                # add to points

                if(len(rgb) > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
                else:
                    points.append([X,Y,Z]);

    return points;


def point_cloud_from_feature_points(disparity, points, rgb=[]):
    points_3d = []

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then get reasonable scaling in X and Y output

    Zmax = ((f * B) / 2);

    for p in points:
        x = int(p[0])
        y = int(p[1])
        #print(x, y)

        # if we have a valid non-zero disparity
        #print(disparity[0])
        if (disparity[y][x] > 0):

            # calculate corresponding 3D point [X, Y, Z]

            # stereo lecture - slide 22 + 25

            Z = (f * B) / disparity[y,x];

            X = ((x - image_centre_w) * Zmax) / f;
            Y = ((y - image_centre_h) * Zmax) / f;

            # add to points_3d

            if(len(rgb) > 0):
                points_3d.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
            else:
                points_3d.append([X,Y,Z]);

    return points_3d

#####################################################################

# project a set of 3D points back the 2D image domain

def project_3D_points_to_2D_image_points(points):

    points2 = [];

    # calc. Zmax as per above

    Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2;

    for i1 in range(len(points)):

        # reverse earlier projection for X and Y to get x and y again

        x = ((points[i1][0] * camera_focal_length_px) / Zmax) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / Zmax) + image_centre_h;
        points2.append([x,y]);

    return points2;

#####################################################################

# resolve full directory location of data set for left / right images

def load_images(img_name, display=False, prnt=False):
    full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
    full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

    full_path_filename_left = os.path.join(full_path_directory_left, img_name);
    full_path_filename_right = (full_path_filename_left.replace("left", "right")).replace("_L", "_R");

    # for sanity print out these filenames
    if prnt:
        print(full_path_filename_left);
        print(full_path_filename_right);

    # check the files actually exist
    imgL = imgR = None
    grayL = grayR = None
    if (os.path.isfile(full_path_filename_left) and os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR);
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR);

        if prnt:
            print("-- files loaded successfully");

        # remember to convert to grayscale (as the disparity matching works on grayscale)
        # N.B. need to do for both as both are 3-channel images

        #grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
        #grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

        if display:
            cv2.imshow("left", grayL)
            cv2.imshow("right", grayR)
            cv2.waitKey(0)
    return imgL, imgR

def compute_disparity(imgL, imgR, display=False):
    # setup the disparity stereo processor to find a maximum of 128 disparity values
    # (adjust parameters if needed - this will effect speed to processing)

    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY);

    max_disparity = 128;
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)
    disparity = stereoProcessor.compute(grayL,grayR);

    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5; # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);

    # display image (scaling it to the full 0->255 range based on the number
    # of disparities in use for the stereo part)

    if display:
        cv2.imshow("disparity", (disparity_scaled * (255. / max_disparity)).astype(np.uint8));
        cv2.waitKey(0)
    return disparity_scaled


def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #cv2.circle(img,(x,y),100,(255,0,0),-1)
        print(x, y)

def find_n_best_feature_points(imgL, imgR, n, thresh=100, draw=False):
    #fast = cv2.FastFeatureDetector_create(threshold=thresh)
    #kpL = fast.detect(imgL, None)
    #kpR = fast.detect(imgR, None)

    

    orb = cv2.ORB_create()
    kpL, desL = orb.detectAndCompute(imgL, None)
    kpR, desR = orb.detectAndCompute(imgR, None)
    #print(list(zip(kpL, desL)))
 
    # Get the keypoint, descriptor pairs in order based off the response value of the keypoint
    """kpL_and_desL = [(x, y) for x, y in sorted(zip(kpL, desL), key= lambda x:x[0].response)]
    kpR_and_desR = [(x, y) for x, y in sorted(zip(kpR, desR), key= lambda x:x[0].response)]

    unzippedL = list(zip(*kpL_and_desL))
    kpL = unzippedL[0]
    desL = np.array(unzippedL[1])

    unzippedR = list(zip(*kpR_and_desR))
    kpR = unzippedR[0]
    desR = np.array(unzippedR[1])
    print(desL)
    #[kpR, desR] = sorted([kpR, desR], key = lambda x:x[1].response)
    """
    bf = cv2.BFMatcher(cv2.HAMMING_NORM_TYPE, crossCheck=True)
    matches = bf.match(desL, desR)

    matches = sorted(matches, key = lambda x:x.distance, reverse=False)
    #print(matches[0].distance, matches[10].distance)
    pointsL = []
    pointsR = []
    for i in matches[:n]:
        if i.distance < thresh:
            pointsL.append(kpL[i.queryIdx]) # First image
            pointsR.append(kpR[i.trainIdx]) # Second image
    #matches = sorted(matches, key = lambda x:x.distance)
    #matches = matches[-n:]
    

    # Sort them in the increasing order of their feature strength
    #kpL = sorted(kpL, key = lambda x:x.response)
    #kpR = sorted(kpR, key = lambda x:x.response)

    """if len(kpL) >= n and len(kpR) >= n:
        # Get the n best
        kpL = kpL[-n:] # Last n elements
        kpR = kpR[-n:]
    else:
        kpL = kpR = [] # Return no keypoints if n too big
    """
    

    #print([(f.pt[0], f.pt[1]) for f in pointsR])
    if draw and imgL is not None:
        #cv2.namedWindow("Matches", cv2.WINDOW_NORMAL)
        #img3 = None
        #img3 = cv2.drawMatches(imgL,kpL,imgR,kpR,matches[:n], img3, flags=2)
        #cv2.imshow("Matches", img3)
        #imgR = cv2.drawKeypoints(imgR, pointsR, imgR, color=(255,0,0))
        #cv2.imshow("KPs right", imgR)
        imgL = cv2.drawKeypoints(imgL, pointsL, imgL, color=(0,255,0))
        cv2.imshow("KPs left", imgL)
        #cv2.setMouseCallback('KPs right',draw_circle)
        cv2.waitKey(1)
    return pointsL, pointsR
    
        


def project_disparity_to_point_cloud(disparity_scaled, img=[], write_to_file=False):
    max_disparity = 128;
    # project to a 3D colour point cloud (with or without colour)

    #points = project_disparity_to_3d(disparity_scaled, max_disparity);
    points = project_disparity_to_3d(disparity_scaled, img);

    # write to file in an X simple ASCII X Y Z format that can be viewed in 3D
    # using the on-line viewer at http://lidarview.com/
    # (by uploading, selecting X Y Z format, press render , rotating the view)
    if write_to_file:
        point_cloud_file = open('3d_points.txt', 'w');
        csv_writer = csv.writer(point_cloud_file, delimiter=' ');
        csv_writer.writerows(points);
        point_cloud_file.close();
    
    return points

        
"""        # select a random subset of the 3D points (4 in total)
        # and them project back to the 2D image (as an example)

        pts = project_3D_points_to_2D_image_points(random.sample(points, 4));
        pts = np.array(pts, np.int32);
        pts = pts.reshape((-1,1,2));

        cv2.polylines(imgL,[pts],True,(0,255,255), 3);

        cv2.imshow('left image',imgL)
        cv2.imshow('right image',imgR)

        # wait for a key press to exit

        cv2.waitKey(0);

    else:
            print("-- files skipped (perhaps one is missing or path is wrong)");
            print();

    # close all windows

    cv2.destroyAllWindows()"""

#####################################################################
