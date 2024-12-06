'''
CS585 Assignment 3 - Part 1
February 26, 2024

Demetrios Kechris
Roger Finnerty
Ben Burnham

'''

import json
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# part 1:
def load_obj_each_frame(data_file):
    with open(data_file, 'r') as file:
        frame_dict = json.load(file)
    return frame_dict


def draw_target_object_center(video_file,obj_centers, vid_name):
    count = 0
    cap = cv.VideoCapture(video_file)
    ok, image = cap.read()
    vidwrite = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'MP4V'), 30, (700,500))
    while ok:
        ######!!!!#######
        image = cv.resize(image, (700, 500)) # make sure your video is resize to this size, otherwise the coords in the data file won't work !!!
        ######!!!!#######
        
        for past in obj_centers[:count]:
            pos_x,pos_y = past
            image = cv.circle(image, (int(pos_x),int(pos_y)), 1, (255,0,0), 2)
        
        pos_x,pos_y = obj_centers[count]
        image = cv.circle(image, (int(pos_x),int(pos_y)), 1, (0,0,255), 2)

        vidwrite.write(image)
        ok, image = cap.read()

        count+=1 
    vidwrite.release()
    

def plot_track(track):
    for pos in track:
        plt.plot(pos[0], pos[1], 'b*')
    for pos in measurements:
        plt.plot(pos[0], pos[1], 'r*')

    plt.show()

# Alpha Beta Function
def run_Alpha_Beta(measurements):
    def predict_step(prev_pos, prev_vel, step=1):
        """Compute prior estimates of position and velocity"""
        pos_prior = prev_pos + step * prev_vel
        vel_prior = prev_vel
        return pos_prior, vel_prior


    def update_step(pos_prior, vel_prior, pos_observed, alpha, beta, step=1):
        """Compute posterior estimates of position and velocity"""
        residual = pos_observed - pos_prior
        pos_posterior = pos_prior + alpha * residual
        vel_posterior = vel_prior + (beta/step) * residual
        return pos_posterior, vel_posterior
    
    # Initialization
    prev_pos = np.array(measurements[1])
    prev_vel = np.array([0,0])
    step = 1
    alpha = 0.85
    beta = .05

    # to be removed later for backwards pass
    track = [[0,0],[prev_pos[0],prev_pos[1]]]

    # Measurements start at second time step
    for measurement in measurements[2:]:
        # Prediction step
        pos_prior, vel_prior = predict_step(prev_pos, prev_vel, step)

        # Update step
        if measurement == [-1, -1]:
            # use prior estimate for position as measurement
            pos_post, vel_post = update_step(pos_prior, vel_prior, pos_prior, alpha, beta, step)
        else:
            pos_post, vel_post = update_step(pos_prior, vel_prior, np.array(measurement), alpha, beta, step)

        # track.append(pos_post))
        track.append([int(round(pos_post[0])), int(round(pos_post[1]))])
        prev_pos = pos_post
        prev_vel = vel_post
    
    return track

'''
Kalman Filter Class Equations

create Kalaman filter class
must have variables A, B, and all arrays
x, xhat, m, k, z

A (Y intercept)
B (Slope)

State equation
x[t] = Ax[t - 1] + Bq[t]

A and B are known p x p by p x r matrices.
assume B = 0

Measurement Equation
z[t] = H[t] x[t] + w[t]
'''

class KMfilter():
    def __init__(self):
        self.step = .05
        self.A = np.array([[1, self.step, 0, 0],[ 0, 1, 0, 0],[ 0, 0, 1, self.step],[ 0, 0 ,0, 1]])
        self.B = 0
        self.H = np.array([[1, 0, 0, 0],[ 0, 0, 1, 0]])
        self.c_w = np.array([0.5]).reshape(1,1)

    # 1. State prediction
    # xhat[t|t − 1] = A xhat[t−1|t−1]
    def xhat(self, xprior):
        newxhat = self.A @ xprior
        # print("newxhat ", newxhat)
        return newxhat

    # 2. MSE Prediction:
    # M[t|t−1] = A M[t−1|t−1]A^T + BCqB^T
    def MSEhat(self, mseprior):
        msepost = self.A @ mseprior @ self.A.T + .05
        # print("msepost ", msepost)
        return msepost

    # 3. Kalman Gain Computation:
    # K[t] = M [t|t − 1]H^T [t] (Cw[t] + H[t] M [t|t − 1]H^T [t])^−1
    def KGC(self, mseprior):
        self.k = mseprior @ self.H.T @ (np.linalg.inv(self.c_w + self.H @ mseprior @ self.H.T))
        # print("KGC ", self.k)
        return self.k

    # 4. State Estimation (= Correction):
    # xhat[t|t] = xhat[t|t − 1] + K[t] (z[t] − H[t] xhat[t|t − 1])
    def xhat_estimate(self, xprior, k, measurement):
        xhat_estimate = xprior + k @ ( measurement - self.H @ xprior )
        # print("state estimate ", xhat_estimate)
        return xhat_estimate

    # 5. MSE Estimation:
    # M[t|t] = (1 − K[t]) H[t] M[t|t − 1]
    def MSE_estimate(self, mseprior, k):
        MSE_estimate = (1 - k ) @ self.H @ mseprior
        return MSE_estimate


def kalman_track(measurements):
    # Initialization
    prev_pos = np.array(measurements[1])
    x_pos_init = prev_pos[0]
    y_pos_init = prev_pos[1]

    prev_pos2 = np.array(measurements[2])
    x_pos_init2 = prev_pos2[0]
    y_pos_init2 = prev_pos2[1]

    x_vel_init = x_pos_init2 - x_pos_init
    y_vel_init = y_pos_init2 - y_pos_init

    prev_state = np.array([[x_pos_init],[0],[y_pos_init],[0]])
    # prev_state = np.array([[x_pos_init],[x_vel_init],[y_pos_init],[y_vel_init]])
    # print("prev_state ", prev_state)

    # Initialize Kalman Filter
    km = KMfilter()

    # Initialize covariance matrix as identity
    mse_post = np.identity(4)

    # Save first position to track
    track = [[int(prev_pos[0]),int(prev_pos[1])]]

    # Measurements start at second time step
    for measurement in measurements[1:]:
        # Prediction step
        state_prior = km.xhat(prev_state)
        mse_prior = km.MSEhat(mse_post)
        k = km.KGC(mse_prior)

        # Update step
        if measurement == [-1, -1]:
            # use prior estimate for position as measurement
            sub_measurement = np.array([state_prior[0],
                                        state_prior[2]])
            state_post = km.xhat_estimate(state_prior, k, sub_measurement)
        else:
            state_post = km.xhat_estimate(state_prior, k, np.array(measurement).reshape(2,1))
        
        mse_post = km.MSE_estimate(mse_prior, k)
        track.append([int(state_post[0]), int(state_post[2])])
        prev_state = state_post

    # Duplicate last point to account for not having the first point
    track.append(track[-1])

    return track
    

track_path = 'object_to_track.json'
frame_dict = load_obj_each_frame(track_path)
measurements = frame_dict['obj']

# track = run_Alpha_Beta(measurements)
track = kalman_track(measurements)
# plot_track(track)

# Save track to json
data = {"obj": track}
with open('part_1_object_tracking.json', 'w') as json_file:
    json.dump(data, json_file)

# Generate video
vid_path = 'commonwealth.mp4'
# out_video = draw_target_object_center(vid_path, track, 'alpha_beta.mp4')
out_video = draw_target_object_center(vid_path, track, 'part_1_demo.mp4')
