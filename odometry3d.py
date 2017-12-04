import stereo_to_3d as s3d 
import random
import cv2
import numpy as np
import os
import mono_stream as ms
import matplotlib.pyplot as plt
import math
import transformations as trans

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres
image_centre_h = 262.0
image_centre_w = 474.5
principle_point = (474.5, 262.0) #Taken from mono_stream.py

camera_matrix = np.array([[camera_focal_length_m, 0, principle_point[0]],
                         [0, camera_focal_length_m, principle_point[1]],
                         [0, 0, 1]])

prev_r = []
prev_t = []


def match_3d_features(points1, points2):
    h_mat = cv2.findHomography(points1, points2, method="RANSAC") 
    _, r, t, _ = cv2.decomposeHomographyMat(h_mat, camera_matrix)
    return r, t

def calculate_t(x, fp1, fp2):
    sum = 0
    x = np.mat()
    for a in fp1:
        for b in fp2:
            sum += abs(a - x*b)
    return sum

def matched_points_from_point_cloud(disparity1, features1, disparity2, features2, rgb=[]):
    points_3d_1 = []
    points_3d_2 = []

    f = camera_focal_length_px
    B = stereo_camera_baseline_m

    height, width = disparity1.shape[:2]

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then get reasonable scaling in X and Y output

    Zmax = ((f * B) / 2)

    for p in range(0, len(features1)):
        p1 = features1[p]
        p2 = features2[p]

        x1 = int(p1[0])
        y1 = int(p1[1])
        
        x2 = int(p2[0])
        y2 = int(p2[1])
        #print(x, y)

        # if we have a valid non-zero disparity
        #print(disparity[0])
        if disparity1[y1][x1] > 0 and disparity2[y2][x2] > 0:

            Z1 = (f * B) / disparity1[y1,x1]
            X1 = ((x1 - image_centre_w) * Zmax) / f
            Y1 = ((y1 - image_centre_h) * Zmax) / f

            # add to points_3d

            if(len(rgb) > 0):
                points_3d_1.append([X1,Y1,Z1,rgb[y1,x1,2], rgb[y1,x1,1],rgb[y1,x1,0]])
            else:
                points_3d_1.append([X1,Y1,Z1])

            Z2 = (f * B) / disparity2[y2,x2]
            X2 = ((x2 - image_centre_w) * Zmax) / f
            Y2 = ((y2 - image_centre_h) * Zmax) / f

            # add to points_3d

            if(len(rgb) > 0):
                points_3d_2.append([X2,Y2,Z2,rgb[y2,x2,2], rgb[y2,x2,1],rgb[y2,x2,0]])
            else:
                points_3d_2.append([X2,Y2,Z2])

    return np.array(points_3d_1), np.array(points_3d_2)


def vector_norm(data, axis=None, out=None):
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def pose_from_point_cloud(img1, img2):
    left, right = s3d.load_images(img1)
    left2, right2 = s3d.load_images(img2)
    matched_features_1, matched_features_2 = s3d.find_n_best_feature_points(left, left2, 300, 40, False)

    features_left_1 = np.array([(f.pt[0], f.pt[1]) for f in matched_features_1])
    features_left_2 = np.array([(f.pt[0], f.pt[1]) for f in matched_features_2])
    #print(features_left_1, features_left_2)

    disparity_1 = s3d.compute_disparity(left, right, False)
    disparity_2 = s3d.compute_disparity(left2, right2, False)
    
    points_3d_1, points_3d_2 = matched_points_from_point_cloud(disparity_1, features_left_1, disparity_2, features_left_2) # MAKE ONE THAT DOES BOTH IMAGES AT SAME TIME, SINCE LOTS OF THE POINTS DO NOT GET DONE DUES TO DISPARITY FOR SOME REASON, SO THE LEFTOVER ONES ARENT MATCHED
    #print(np.transpose(points_3d_1))
    #formatted_points_1 = formatted_points_2 = []
    #for p in points_3d_1:
        
    _, transformation, _ = cv2.estimateAffine3D(points_3d_1, points_3d_2, ransacThreshold=0.97)
    #RESHAPE POINT ARRAYS
    #print(points_3d_1, points_3d_2)
    #affine3d = trans.affine_matrix_from_points(np.transpose(points_3d_1), np.transpose(points_3d_2), False, False)
    #print(transformation, affine3d)
    transformation = np.append(transformation, [[0.0, 0.0, 0.0, 1.0]], axis=0)
    #print(transformation, affine3d)
    scale, shear, angles, translation, perspective = trans.decompose_matrix(transformation)

    r = angles
    t = translation
    """
    h_mat, _ = cv2.findHomography(points_3d_1, points_3d_2, method=cv2.RANSAC) 
    _, r, t, _ = cv2.decomposeHomographyMat(h_mat, camera_matrix)
    print("r, t:", r, t)
    global prev_r, prev_t
    if len(prev_r) > 0 and len(prev_t) > 0:
        best_r = r[0]
        for i in r[1:]:
            #print(i)
            if abs(prev_r[0][0] - i[0][0]) < abs(prev_r[0][0] - best_r[0][0]):
                best_r = i

        best_t = t[0]
        for i in t[1:]:
            if abs(prev_t[0][0] - i[0][0]) < abs(prev_t[0][0] - best_t[0][0]):
                best_t = i

    else:
        best_r = r[0]
        best_t = t[0]
        for i in r:
            if i[0][0] > 0.99:
                best_r = i
        for i in t:
            if i[2] > 0:
                best_t = i
        
    prev_r = best_r
    prev_t = best_t"""

    return r, t


def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def run_odometry():
    num_photos_to_do = 3000
    master_path_to_dataset = "TTBB-durham-02-10-17-sub5" # ** need to edit this **
    directory_to_cycle = "left-images"     # edit this for left or right image set
    full_path_directory = os.path.join(master_path_to_dataset, directory_to_cycle)
    images = sorted(os.listdir(full_path_directory))
    gps, imu = ms.load_gps_and_imu_data(master_path_to_dataset)

    prev_prev_image = images[0]
    prev_image = images[1]
    r, t = pose_from_point_cloud(prev_prev_image, prev_image)
    #print(t*0.001)
    r_sum, t_sum = r, t

    gps_1 = ms.get_gps_data(0, gps)
    gps_2 = ms.get_gps_data(1, gps)
    gps_distance = ms.gps_to_meters(gps_1[0], gps_2[0], gps_1[1], gps_2[1])
    starting_photo_index = gps_counter = 2

    initial_gps_pos = gps_1

    calculated_pos = np.array(initial_gps_pos)
    calculated_angle = r[1]+np.pi/2#rotation_matrix_to_euler_angles(r)[1]+np.pi/2

    total_distance = np.sqrt((t[0] * t[0]) + (t[2] * t[2]))
    for curr_image in images[starting_photo_index:num_photos_to_do]:
        r, t = pose_from_point_cloud(prev_image, curr_image)
        print(r[1], t)
        #print(r, t*0.0001)
        if True:#(r is not None) and (t is not None) and not math.isnan(r[0][0]):
            #t_sum = t_sum + np.dot(r_sum, t)
            #r_sum = np.dot(r, r_sum)

            distance_travelled = np.sqrt((t[0] * t[0]) + (t[2] * t[2]))
            total_distance += distance_travelled

            gps_1 = ms.get_gps_data(gps_counter-1, gps)
            gps_2 = ms.get_gps_data(gps_counter, gps)
            gps_counter += 1
            gps_distance += ms.gps_to_meters(gps_1[0], gps_2[0], gps_1[1], gps_2[1])

            calculated_angle += r[1] #rotation_matrix_to_euler_angles(r)[1]
            
            #scale = math.sqrt(sx**2 + sy**2)
            #if scale > 10:
            #    scale = 1
            scale = 1

            calculated_pos[0] += distance_travelled*math.cos(calculated_angle)*0.00001*scale
            calculated_pos[1] += distance_travelled*math.sin(calculated_angle)*0.00001*scale

            plt.plot(gps_2[0], gps_2[1], 'ro')
            plt.plot(calculated_pos[0], calculated_pos[1], 'go')

            prev_image = curr_image

            plt.pause(0.01)

run_odometry()