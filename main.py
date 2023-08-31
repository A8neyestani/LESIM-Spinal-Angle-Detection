"""
@param self
LESIM Project
UNIVERSITÀ DEGLI STUDI DEL SANNIO Benevento
Authors: Arman Neyestani,
         Francesco Picariello
Title: Detect and measuring the Angles in Order to spinal cord.
"""

# Import required libraries and modules
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
from shapely.geometry import LineString, Point
import cofunctions as co
import time
import pandas as pd

# Initialize the video source (can be a file or camera device)
cap = cv2.VideoCapture('tes1_trial_0.mp4')
angleLeft_list = []
angleRight_list = []
recorded_time = []

# Define a VideoWriter object to save the output
out = cv2.VideoWriter('tes1.mp4', -1, 29, (640, 480))
num_frames = 0
start_time = time.time()

# Initialize the mediapipe pose model with given confidence thresholds
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Convert frame color from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image to detect poses
        results = pose.process(image)

        # Convert frame color back from RGB to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract pose landmarks from the results
        try:
            landmarks = results.pose_landmarks.landmark

            # Extract required landmark coordinates
            RIGHT_HIP = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            LEFT_HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            RIGHT_SHOULDER = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            LEFT_SHOULDER = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            LEFT_KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            RIGHT_KNEE = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # Compute the spinal line and find intersections with hips
            spin = co.spin_line(LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
            intersection_point_left = co.find_intersection(spin, [LEFT_HIP, LEFT_KNEE])
            intersection_point_right = co.find_intersection(spin, [RIGHT_HIP, RIGHT_KNEE])

            # Compute the angles using the spinal line and intersections
            angleLeft = co.calculate_angle(spin[1], [intersection_point_left[0], intersection_point_left[1]], LEFT_HIP)
            angleRight = co.calculate_angle(spin[1], [intersection_point_right[0], intersection_point_right[1]], RIGHT_HIP)

            # Save the computed angles and time
            angleLeft_list.append(angleLeft)
            angleRight_list.append(angleRight)
            recorded_time.append(time.time() - start_time)

            # Visualize the results on the image
            # Display angles, time, FPS, and lab information
            cv2.putText(image, 'Left Angle(degree): ' + str(angleLeft)[:6], (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, 'Right Angle(degree): ' + str(angleRight)[:6], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, 'Time(s): ' + str(time.time() - start_time)[:6], (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, 'Rate(FPS): ' + str(num_frames / (time.time() - start_time))[:6], (400, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, 'LESIM LAB, Sannio University, Benevento, Italy ', (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # Draw the calculated lines and intersection points on the image
            cv2.line(image, tuple(np.multiply(spin[0], [640, 480]).astype(int)), tuple(np.multiply(spin[1], [640, 480]).astype(int)), (0, 255, 0), 4)
            cv2.line(image, tuple(np.multiply(LEFT_KNEE, [640, 480]).astype(int)), tuple(np.multiply(LEFT_HIP, [640, 480]).astype(int)), (255, 0, 0), 4)
            cv2.line(image, tuple(np.multiply(LEFT_HIP, [640, 480]).astype(int)), tuple(np.multiply(intersection_point_left, [640, 480]).astype(int)), (255, 0, 0), 1)
            cv2.line(image, tuple(np.multiply(intersection_point_left, [640, 480]).astype(int)), tuple(np.multiply(spin[1], [640, 480]).astype(int)), (255, 0, 0), 1)
            cv2.line(image, tuple(np.multiply(RIGHT_KNEE, [640, 480]).astype(int)), tuple(np.multiply(RIGHT_HIP, [640, 480]).astype(int)), (0, 0, 255), 4)
            cv2.line(image, tuple(np.multiply(RIGHT_HIP, [640, 480]).astype(int)), tuple(np.multiply(intersection_point_right, [640, 480]).astype(int)), (0, 0, 255), 1)
            cv2.line(image, tuple(np.multiply(intersection_point_right, [640, 480]).astype(int)), tuple(np.multiply(spin[1], [640, 480]).astype(int)), (0, 0, 255), 1)
            cv2.circle(image, tuple(np.multiply(intersection_point_left, [640, 480]).astype(int)), 4, (255, 0, 0), -1)
            cv2.circle(image, tuple(np.multiply(intersection_point_right, [640, 480]).astype(int)), 4, (0, 0, 255), -1)

        except:
            pass

        # Draw landmarks on the image
        # if results.pose_landmarks:
        #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Update the number of processed frames
        num_frames += 1

        # Write the resulting frame to output video
        out.write(image)

        # Display the resulting frame
        cv2.imshow('Mediapipe Feed', image)

        # Break loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Close the video stream
cap.release()
out.release()
cv2.destroyAllWindows()

# Save the computed angles and time in a CSV file
df = pd.DataFrame(list(zip(recorded_time, angleLeft_list, angleRight_list)), columns=['Time(s)', 'Left Angle(degree)', 'Right Angle(degree)'])
df.to_csv('angles.csv', index=False)
