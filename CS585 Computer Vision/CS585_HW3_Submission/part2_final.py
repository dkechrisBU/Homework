'''
CS585 Assignment 3 - Part 2
February 26, 2024

Demetrios Kechris
Roger Finnerty
Ben Burnham

'''

import json
import cv2 as cv
import numpy as np
from kalman import KMfilter
import scipy
import copy


def load_obj_each_frame(data_file):
    with open(data_file, 'r') as file:
      frame_dict = json.load(file)
    return frame_dict

# Visualizes known cars and adds identifier to each on every frame
def draw_car(car,image):
    x = int(car[1])
    y = int(car[2])

    id = car[0]

    # Initialize cv.putText
    text = str(id)
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (x, y)
    font_scale = 0.5
    color = (0, 0, 255)  # blue in BGR format
    thickness = 2

    # Add the text to the image
    image = cv.putText(image, text, position, font, font_scale, color, thickness)
    return cv.circle(image, (x,y), radius=35, color=(0, 0, 255), thickness=1)


def draw_object(object_dict,image,color = (0, 255, 0), thickness = 2,c_color= (255, 0, 0)):
    # draw box
    x = object_dict['x_min']
    y = object_dict['y_min']
    width = object_dict['width']
    height = object_dict['height']
    image = cv.rectangle(image, (x, y), (x + width, y + height), color, thickness)

    #####################################
    # Add ID to image:
    # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/

    id = object_dict['id']

    # Initialize cv.putText
    text = str(id)
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (x, y)
    font_scale = 1
    color = (255, 0, 0)  # blue in BGR format
    thickness = 2

    # Add the text to the image
    cv.putText(image, text, position, font, font_scale, color, thickness)

    #####################################
    return image


def draw_objects_in_video(video_file,frame_dict,car_frame_data):
    print(len(car_frame_data))
    print(len(frame_dict))
    count = 0
    cap = cv.VideoCapture(video_file)
    frames = []
    ok, image = cap.read()
    vidwrite = cv.VideoWriter("part_2_demo.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700,500))
    while ok:
        ######!!!!#######
        image = cv.resize(image, (700, 500)) # make sure your video is resize to this size, otherwise the coords in the data file won't work !!!
        ######!!!!#######
        obj_list = frame_dict[str(count)]
        for obj in obj_list:
          image = draw_object(obj,image)
        
        # Draw known cars
        for car_frame in car_frame_data[count]:
            image = draw_car(car_frame,image)

        vidwrite.write(image)
        count+=1
        ok, image = cap.read()
    vidwrite.release()

def obj_assign(frame_dict):
    unique_id = 0
    car_history = []
    threshold = 35
    km = KMfilter()
    car_frame_data = []

    # Iterate through all frames
    for frame in range(len(frame_dict)):

        # Give IDs to objects in first frame
        if frame == 0:
            frame_dict_id = frame_dict[str(frame)]
            for obj in range(len(frame_dict_id)):
                # Save new ID to JSON array
                frame_dict_id[obj]['id'] = unique_id

                # Get objects x and y to initialize new car
                x = frame_dict_id[obj]['x_min'] + (frame_dict_id[obj]['width'] / 2)
                y = frame_dict_id[obj]['y_min'] + (frame_dict_id[obj]['height'] / 2)

                # Create new car for the ID
                car_history.append([unique_id,
                                   x,
                                   y,
                                   frame,   # 3 Frame
                                   0,       # 4 Last dist
                                   0,       # 5 Last obs
                                   np.array([[x],[0],[y],[0]]),       # 6 state_prior
                                   np.identity(4),  # 7 mse_prior
                                   0])       # 8 kalman gain

                unique_id += 1  # step ID counter

        # Evaluate from second frame on
        else:
            # Run prediction for all known cars
            for i, car in enumerate(car_history):
                car_history[i][6] = km.xhat(car[6])     # state_prior
                car_history[i][7] = km.MSEhat(car[7])   # mse_prior
                car_history[i][8] = km.KGC(car[7])      # kalman gain

            # Get distance from every object to every known car
            obj_list = frame_dict[str(frame)]
            delta = np.zeros((len(obj_list), len(car_history)))
            for obj in range(len(obj_list)):
                x = obj_list[obj]['x_min'] + (obj_list[obj]['width'] / 2)
                y = obj_list[obj]['y_min'] + (obj_list[obj]['height'] / 2)
                for id, car in enumerate(car_history):
                    # compute delta for every object / ID pair
                    delta[obj,id] = np.sqrt((x-car[6][0])**2+(y-car[6][2])**2)
                    #print(delta[obj,id])

            # Create an nxn cost Matrix
            row_delta, col_delta = np.shape(delta)
            if row_delta > col_delta:
                # adding new car
                ones_array = 900*np.ones((row_delta, row_delta - col_delta))
                delta = np.hstack((delta, ones_array))
            elif row_delta < col_delta:
                # adding fake object
                ones_array = 1000*np.ones((col_delta - row_delta, col_delta))
                delta = np.vstack((delta, ones_array))
            #print(delta)

            # Compute Hungarian assigment
            obj_ind, car_ind = scipy.optimize.linear_sum_assignment(delta)
            print("frame",frame, obj_ind, car_ind)

            # Assign IDs based on result
            for obj in obj_ind:
                # print(delta[obj,car_ind[obj]])

                # within threshold, tagging with id and updating km filter
                if delta[obj,car_ind[obj]] < threshold:
                    # Get ID
                    car = car_history[car_ind[obj]]

                    # Save ID to JSON array
                    frame_dict[str(frame)][obj]['id'] = car[0]

                    # Get x and y for object (measurement)
                    x = frame_dict[str(frame)][obj]['x_min'] + (frame_dict[str(frame)][obj]['width']/2)
                    y = frame_dict[str(frame)][obj]['y_min'] + (frame_dict[str(frame)][obj]['height']/2)

                    # Update kalman filter
                    car[6] = km.xhat_estimate(car[6], car[8], np.array([[x],[y]]))
                    car[7] = km.MSE_estimate(car[7], car[8])
                    car[1] = car[6][0]
                    car[2] = car[6][2]

                    # Save update to car history array
                    car_history[car_ind[obj]] = car

                # Spare car, update kalman filter
                elif delta[obj,car_ind[obj]] == 1000:
                    # Get ID
                    car = car_history[car_ind[obj]]

                    # Use prediction as measurement
                    x = car[6][0]
                    y = car[6][2]

                    # Update kalman filter
                    car[6] = km.xhat_estimate(car[6], car[8], np.array([x,y]))
                    car[7] = km.MSE_estimate(car[7], car[8])
                    car[1] = car[6][0]
                    car[2] = car[6][2]

                    # Save update to car history array
                    car_history[car_ind[obj]] = car

                # Exceeds threshold, create new car
                elif delta[obj,car_ind[obj]] >= threshold:
                    # Save new ID to JSON array
                    frame_dict[str(frame)][obj]['id'] = unique_id

                    # Get objects x and y to initialize new car
                    x = frame_dict[str(frame)][obj]['x_min'] + (frame_dict[str(frame)][obj]['width']/2)
                    y = frame_dict[str(frame)][obj]['y_min'] + (frame_dict[str(frame)][obj]['height']/2)

                    # Create new car for the ID
                    car_history.append([unique_id,
                                        x,
                                        y,
                                        frame,   # 3 Frame
                                        0,       # 4 Last dist
                                        0,       # 5 Last obs
                                        np.array([[x],[0],[y],[0]]),       # 6 state_prior
                                        np.identity(4),  # 7 mse_prior
                                        0])      # 8 kalman gain
                    
                    # step ID counter
                    unique_id +=1

            """
            #greedy part

            delta = np.zeros((len(obj_list) * len(last_known),3))

            # run comparison for everyobject in the current frame
            for obj in range(len(obj_list)):
                min_delta = threshold
                x = obj_list[obj]['x_min']
                y = obj_list[obj]['y_min']

                # compare object to last known pos of every id
                for id, car in enumerate(last_known):
                    # compute delta

                    delta[obj*len(last_known) + id] = np.array([np.sqrt((x-car[1])**2+(y-car[2])**2), obj, id])

                    # update if new min is found
                    # if delta < min_delta:
                    #     min_delta = delta
                    #     obj['id'] = id
            sorted_delta = delta[delta[:, 0].argsort()]
            new_delta = np.zeros((len(obj_list),1))

            #index of new_delta is object to be matched, value is car (if applicable)
            # if val == -1, new track to be created
            new_delta = new_delta - 1
            whitecar = np.ones((len(last_known),1))
            whiteobj = np.ones((len(obj_list),1))

            for i in sorted_delta:
                if i[0] > threshold:
                    break
                #print("i ", i)
                if whiteobj[int(i[1])] * whitecar[int(i[2])] == 1:
                    new_delta[int(i[1])] = i[2]
                    whiteobj[int(i[1])] = 0
                    whitecar[int(i[2])] = 0
                    # Get ID
                    car = car_history[car_ind[obj]]

                    # Save ID to JSON array
                    frame_dict[str(frame)][obj]['id'] = car[0]

                    # Get x and y for object (measurement)
                    x = frame_dict[str(frame)][obj]['x_min'] + (frame_dict[str(frame)][obj]['width']/2)
                    y = frame_dict[str(frame)][obj]['y_min'] + (frame_dict[str(frame)][obj]['height']/2)

                    # Update kalman filter
                    car[6] = km.xhat_estimate(car[6], car[8], np.array([[x],[y]]))
                    car[7] = km.MSE_estimate(car[7], car[8])
                    car[1] = car[6][0]
                    car[2] = car[6][2]

                    # Save update to car history array
                    car_history[car_ind[obj]] = car

            for i in whitecar:
                if i == 1:
                    car = car_history[car_ind[obj]]

                    # Use prediction as measurement
                    x = car[6][0]
                    y = car[6][2]

                    # Update kalman filter
                    car[6] = km.xhat_estimate(car[6], car[8], np.array([x,y]))
                    car[7] = km.MSE_estimate(car[7], car[8])
                    car[1] = car[6][0]
                    car[2] = car[6][2]

                    # Save update to car history array
                    car_history[car_ind[obj]] = car

            for i in whiteobj:
                if i == 1:
                    # Save new ID to JSON array
                    frame_dict[str(frame)][obj]['id'] = unique_id

                    # Get objects x and y to initialize new car
                    x = frame_dict[str(frame)][obj]['x_min'] + (frame_dict[str(frame)][obj]['width']/2)
                    y = frame_dict[str(frame)][obj]['y_min'] + (frame_dict[str(frame)][obj]['height']/2)

                    # Create new car for the ID
                    car_history.append([unique_id,
                                        x,
                                        y,
                                        frame,   # 3 Frame
                                        0,       # 4 Last dist
                                        0,       # 5 Last obs
                                        np.array([[x],[0],[y],[0]]),       # 6 state_prior
                                        np.identity(4),  # 7 mse_prior
                                        0])      # 8 kalman gain

                    # step ID counter
                    unique_id +=1
            """
        # capture car states every frame to track movement
        car_frame_data.append(copy.deepcopy(car_history))

        # DEBUG
        #for car in car_history:
            #print('id:',car[0],'x:',car[1],'y:',car[2])

    # Return JSON array and car data
    return frame_dict, car_frame_data


frame_dict = load_obj_each_frame("frame_dict.json")
frame_dict_new, car_frame_data = obj_assign(frame_dict)
video_file = "commonwealth.mp4"
draw_objects_in_video(video_file,frame_dict_new, car_frame_data)

with open('part_2_frame_dict.json', 'w') as json_file:
  json.dump(frame_dict_new, json_file)
