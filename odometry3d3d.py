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

camera_matrix = np.array([[camera_focal_length_px, 0, principle_point[0]],
                         [0, camera_focal_length_px, principle_point[1]],
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


def points_3d_2d(disparity, points1, points2, rgb=[]):
    points_3d = [] #points1 go to 3d
    points_2d = [] #points2 stay as 2d, but need to be reduced if some from points1 do not have depth value

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then get reasonable scaling in X and Y output

    Zmax = ((f * B) / 2);

    for p_idx in range(0, len(points1)):
        p = points1[p_idx]
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
                points_3d.append((X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]));
            else:
                points_3d.append((X,Y,Z));
            points_2d.append(points2[p_idx])
    return points_3d, points_2d


def pose_from_point_cloud(img1, img2):
    left, right = s3d.load_images(img1)
    left2, right2 = s3d.load_images(img2)
    matched_features_1, matched_features_2 = s3d.find_n_best_feature_points(left, left2, 500, 100, True, True, 5, 3, 0.4)

    features_left_1 = np.array([(f.pt[0], f.pt[1]) for f in matched_features_1])
    features_left_2 = np.array([(f.pt[0], f.pt[1]) for f in matched_features_2])
    #print(features_left_1, features_left_2)
    
    stereo = cv2.StereoBM_create()
    #stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
    disparity_1 = stereo.compute(left, right)
    disparity_2 = stereo.compute(left2, right2)
    #disparity_1 = s3d.compute_disparity(left, right, False)
    #disparity_2 = s3d.compute_disparity(left2, right2, False)
    
    points_3d_1, points_3d_2 = matched_points_from_point_cloud(disparity_1, features_left_1, disparity_2, features_left_2) # MAKE ONE THAT DOES BOTH IMAGES AT SAME TIME, SINCE LOTS OF THE POINTS DO NOT GET DONE DUES TO DISPARITY FOR SOME REASON, SO THE LEFTOVER ONES ARENT MATCHED
        

    #3D to 2D:
    #points_3d, points_2d = points_3d_2d(disparity_1, features_left_1, features_left_2)
    #_, r, t, _ = cv2.solvePnPRansac(np.array(points_3d, dtype="double"), np.array(points_2d, dtype="double"), camera_matrix, np.array([]))
    #print(r, t)
    #r = rotation_matrix_to_euler_angles(r)
    #r = r[1][0]
    #t = t[2][0] 
    if len(points_3d_1) > 0:
        a = 1
        #Option 1: OpenCV affine 3D
        #_, transformation, _ = cv2.estimateAffine3D(points_3d_1, points_3d_2, ransacThreshold=0.97)
        #transformation = np.append(transformation, [[0.0, 0.0, 0.0, 1.0]], axis=0)

        #Option 2: Trans affine 3D
        #transformation = trans.affine_matrix_from_points(np.transpose(points_3d_1), np.transpose(points_3d_2), False, False)
        
        #Option 3: OpenCV fundamental matrix
        #f, _ = cv2.findFundamentalMat(points_3d_1, points_3d_2)
        #transformation = np.append(np.transpose(transformation), [[0.0, 0.0, 0.0]], axis=0)
        #transformation = np.append(np.transpose(transformation), [[0.0, 0.0, 0.0, 1.0]], axis=0)
        # e = np.dot(np.dot(np.transpose(camera_matrix), f), camera_matrix)
        # w, u, vt = cv2.SVDecomp(e)
        # r = np.dot(np.transpose(np.dot(np.transpose(u), w)) , vt)[0] % np.pi*2
        # t = np.transpose(u)[1]
        #_, r, t, _ = cv2.recoverPose(e, features_left_1, features_left_2, camera_matrix)
        #r = rotation_matrix_to_euler_angles(r)


        #Option 4: OpenCV homography
        # transformation, _ = cv2.findHomography(points_3d_1, points_3d_2)
        # _, rh, th, _ = cv2.decomposeHomographyMat(transformation, camera_matrix)
        # r = rh[0]
        # t = th[0]
        # for i in rh:
        #     if i[0][0] > 0:
        #         r = i
        # for i in th:
        #     if i[2] > 0:
        #         t = i

        #Option 5: Fundamental to essential - http://www.morethantechnical.com/2012/02/07/structure-from-motion-and-3d-reconstruction-on-the-easy-in-opencv-2-3-w-code/ 
        f, _ = cv2.findFundamentalMat(points_3d_1, points_3d_2, cv2.RANSAC)
        e = np.dot(np.dot(np.transpose(camera_matrix), f), camera_matrix)
        #print(f, e)
        #_, r, t, _ = cv2.recoverPose(e, features_left_1, features_left_2, camera_matrix)
        #r = rotation_matrix_to_euler_angles(r)[1]
        #t = t[2]
        w, u, vt = cv2.SVDecomp(e)
        r = np.dot(np.transpose(np.dot(np.transpose(u), w)), vt)
        t = np.transpose(u)
        print(r, t)
        #Option 6: Get 3 essential matrices from doing pairs of points e.g. xy, xz, yz, then average essential mat
        

        #r = rotation_matrix_to_euler_angles(transformation[0:3, 0:3])
        #t = np.transpose(transformation)[3][0:3]
        #r = rotation_matrix_to_euler_angles(r_mat)
        #scale, shear, angles, translation, perspective = trans.decompose_matrix(transformation)
        #r = angles[1]
        #t = translation[2]
    else:
        print("NO FEATURES")
        r = 0 # [0, 0, 0]
        t = 0 #[0, 0, 0]



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
    calculated_angle = r+np.pi/2#[1]+np.pi/2#rotation_matrix_to_euler_angles(r)[1]+np.pi/2

    total_distance = abs(t)#np.sqrt((t[0] * t[0]) + (t[2] * t[2]))
    for curr_image in images[starting_photo_index:num_photos_to_do]:
        r, t = pose_from_point_cloud(prev_image, curr_image)
        print(r, t)
        #print(r, t*0.0001)
        if True: #(r is not None) and (t is not None) and not math.isnan(r[0]):
            #t_sum = t_sum + np.dot(r_sum, t)
            #r_sum = np.dot(r, r_sum)

            distance_travelled = t#[2]#np.sqrt((t[0] * t[0]) + (t[2] * t[2]))
            if distance_travelled > 50:
                distance_travelled = 50
            total_distance += distance_travelled

            gps_1 = ms.get_gps_data(gps_counter-1, gps)
            gps_2 = ms.get_gps_data(gps_counter, gps)
            gps_counter += 1
            gps_distance += ms.gps_to_meters(gps_1[0], gps_2[0], gps_1[1], gps_2[1])

            if abs(r) < 0.5:
                calculated_angle -= r#[1] #rotation_matrix_to_euler_angles(r)[1]
            
            #scale = math.sqrt(sx**2 + sy**2)
            #if scale > 10:
            #    scale = 1
            scale = 1
            calculated_pos[0] += distance_travelled*math.cos(calculated_angle)*0.00001*scale
            calculated_pos[1] += distance_travelled*math.sin(calculated_angle)*0.00001*scale

            plt.plot(gps_2[1], gps_2[0], 'r.')
            plt.plot(calculated_pos[1], calculated_pos[0], 'g.')

            prev_image = curr_image

            plt.pause(0.01)

run_odometry()