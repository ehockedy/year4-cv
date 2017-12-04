import stereo_to_3d as s3d 
import random
import cv2
import numpy as np
import os
import mono_stream as ms
import matplotlib.pyplot as plt
import math


def pose_from_feature_points(img1, img2):
    left, right = s3d.load_images(img1)
    left2, right2 = s3d.load_images(img2)
    matched_features_1, matched_features_2 = s3d.find_n_best_feature_points(left, left2, 100, 100, True)

    features_left_alt = np.array([(f.pt[0], f.pt[1]) for f in matched_features_1])
    features_left_2_alt = np.array([(f.pt[0], f.pt[1]) for f in matched_features_2])
    #print(features_left_2_alt, "\n", features_left_alt, "\n\n\n")

    R = t = None
    scale_x = scale_y = 1
    if len(matched_features_1) > 0 and len(matched_features_2):
        feature_1_diff_x = abs(features_left_alt[0][0] - features_left_alt[1][0])
        feature_2_diff_x = abs(features_left_2_alt[0][0] - features_left_2_alt[1][0])

        feature_1_diff_y = abs(features_left_alt[0][1] - features_left_alt[1][1])
        feature_2_diff_y = abs(features_left_2_alt[0][1] - features_left_2_alt[1][1])

        if feature_2_diff_x > 0:
            scale_x = feature_1_diff_x / feature_2_diff_x
        if feature_2_diff_y > 0:
            scale_y = feature_1_diff_y / feature_2_diff_y

        E, _ = cv2.findEssentialMat(features_left_alt, features_left_2_alt, s3d.camera_matrix)
        _, R, t, _ = cv2.recoverPose(E, features_left_alt, features_left_2_alt, s3d.camera_matrix)
    return R, t, scale_x, scale_y


# Useful functions:
# - solvePnPRansac
# - correctMatches

# Questions
# - If getting point cloud from feature points, do you base it off the feature positions in the left or right image? 
# - Use features from left images, project to 3d point clouds

# Things to think about
# - focal length makes difference if in m and px, m seems to give most reasonable answer

def gps_points_angle(gps_1, gps_2):
    return np.arctan((gps_1[0] - gps_2[0]) / (gps_1[1] - gps_2[1]))


# Source - https://www.learnopencv.com/rotation-matrix-to-euler-angles/ 
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
    r, t, _, _ = pose_from_feature_points(prev_prev_image, prev_image)

    r_sum, t_sum = r, t

    gps_1 = ms.get_gps_data(0, gps)
    gps_2 = ms.get_gps_data(1, gps)
    gps_distance = ms.gps_to_meters(gps_1[0], gps_2[0], gps_1[1], gps_2[1])
    starting_photo_index = gps_counter = 2

    initial_gps_pos = gps_1

    calculated_pos = np.array(initial_gps_pos)
    calculated_angle = rotation_matrix_to_euler_angles(r)[1]+np.pi/2

    total_distance = np.sqrt((t[0] * t[0]) + (t[2] * t[2]))
    for curr_image in images[starting_photo_index:num_photos_to_do]:
        r, t, sx, sy = pose_from_feature_points(curr_image, prev_image)

        if (r is not None) and (t is not None):
            t_sum = t_sum + np.dot(r_sum, t)
            r_sum = np.dot(r, r_sum)

            distance_travelled = np.sqrt((t[0] * t[0]) + (t[2] * t[2]))
            total_distance += distance_travelled

            gps_1 = ms.get_gps_data(gps_counter-1, gps)
            gps_2 = ms.get_gps_data(gps_counter, gps)
            gps_counter += 1
            gps_distance += ms.gps_to_meters(gps_1[0], gps_2[0], gps_1[1], gps_2[1])

            calculated_angle += rotation_matrix_to_euler_angles(r)[1]
            
            scale = math.sqrt(sx**2 + sy**2)
            if scale > 10:
                scale = 1

            calculated_pos[0] += distance_travelled*math.cos(calculated_angle)*0.00001*scale
            calculated_pos[1] += distance_travelled*math.sin(calculated_angle)*0.00001*scale

            plt.plot(gps_2[0], gps_2[1], 'ro')
            plt.plot(calculated_pos[0], calculated_pos[1], 'go')

            prev_image = curr_image

            plt.pause(0.01)

run_odometry()


# Things too add:
# Only dominant forward motion