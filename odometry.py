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
    matched_features_1, matched_features_2 = s3d.find_n_best_feature_points(left, left2, 500, 40, True, binning=True, binsx=3, binsy=3, cutoff=0.4) # Matche between first left image and second left image

    features_left_alt = np.array([(f.pt[0], f.pt[1]) for f in matched_features_1])
    features_left_2_alt = np.array([(f.pt[0], f.pt[1]) for f in matched_features_2])
    
    #matched_left_1, matched_right_1 = s3d.find_n_best_feature_points(left, right, 300, 100, False) # Match between first left image and first right image
    #matched_left_2, matched_right_2 = s3d.find_n_best_feature_points(left2, right2, 300, 100, False) # Match between second left image and second right image

    R = t = None
    scale_x = scale_y = 1
    if len(matched_features_1) > 0 and len(matched_features_2) > 0:
        scale_x = scale_y = 0
        feature_counter_x = feature_counter_y = 0
        

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
    num_photos_to_do = 4000
    master_path_to_dataset = "TTBB-durham-02-10-17-sub5" # ** need to edit this **
    directory_to_cycle = "left-images"     # edit this for left or right image set
    full_path_directory = os.path.join(master_path_to_dataset, directory_to_cycle)
    images = sorted(os.listdir(full_path_directory))
    gps, imu = ms.load_gps_and_imu_data(master_path_to_dataset)

    prev_prev_image = images[0]
    prev_image = images[1]
    r, t, _, _ = pose_from_feature_points(prev_prev_image, prev_image)

    r_sum, t_sum = r, t

    starting_photo_index = gps_counter = 1570#2
    gps_1 = ms.get_gps_data(starting_photo_index, gps)
    gps_2 = ms.get_gps_data(starting_photo_index+1, gps)
    gps_distance = ms.gps_to_meters(gps_1[0], gps_2[0], gps_1[1], gps_2[1])
    

    initial_gps_pos = gps_1

    calculated_pos = np.array(initial_gps_pos)
    calculated_angle = rotation_matrix_to_euler_angles(r)[1]+np.pi/2

    total_distance = np.sqrt((t[0] * t[0]) + (t[2] * t[2]))
    for curr_image in images[starting_photo_index:num_photos_to_do]:
        try:
            r, t, sx, sy = pose_from_feature_points(curr_image, prev_image)
            if (r is not None) and (t is not None):
                #t_sum = t_sum + np.dot(r_sum, t)
                #r_sum = np.dot(r, r_sum)

                distance_travelled = t[2] #np.sqrt((t[0] * t[0]) + (t[2] * t[2]))
                total_distance += distance_travelled

                gps_1 = ms.get_gps_data(gps_counter-1, gps)
                gps_2 = ms.get_gps_data(gps_counter, gps)
                gps_counter += 1
                gps_distance += ms.gps_to_meters(gps_1[0], gps_2[0], gps_1[1], gps_2[1])

                ang_change = rotation_matrix_to_euler_angles(r)[1]
                if abs(ang_change) < 0.4:
                    calculated_angle += rotation_matrix_to_euler_angles(r)[1]
                
                #scale = math.sqrt(sx**2 + sy**2)
                #if scale > 10:
                scale = 2

                calculated_pos[0] += distance_travelled*math.cos(calculated_angle)*0.00001*scale
                calculated_pos[1] += distance_travelled*math.sin(calculated_angle)*0.00001*scale

                plt.plot(gps_2[1], gps_2[0], 'ro')
                plt.plot(calculated_pos[1], calculated_pos[0], 'go')

                prev_image = curr_image

                plt.pause(0.01)
        except Exception:
            print(curr_image)

run_odometry()


# Things too add:
# Only dominant forward motion

#Things to do:
# Quantitive results - difference between gps and vo, then find sum of all differences
# Record output
# Get 3d working...
# Finish report
# Scaling - used 3d data, or use x and y like Tom did
# Restructure code
# Visualise onto map or satelite imagery

#Things to talk about
# Removing lower FPs seems better
# Having stricted matching seems better - the imges change too quickly so seems like not going very fast down road past palatine, as features of lines matched with different lines
# Suitability of this approac for use in actual driving application
# Binning might work better if framerate was higher
# VO outperforms gps - start from image 1570