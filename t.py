import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import pyttsx3
import mediapipe as mp

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize the face mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh_module = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize servos
servo_pin_horizontal = 32  # GPIO pin for horizontal servo
servo_pin_vertical = 33    # GPIO pin for vertical servo
GPIO.setmode(GPIO.BOARD)   # Set GPIO numbering mode
GPIO.setup(servo_pin_horizontal, GPIO.OUT)  # Set horizontal servo pin as an output
GPIO.setup(servo_pin_vertical, GPIO.OUT)    # Set vertical servo pin as an output
servo_horizontal = GPIO.PWM(servo_pin_horizontal, 50)  # Create PWM instance for horizontal servo with frequency 50 Hz
servo_vertical = GPIO.PWM(servo_pin_vertical, 50)      # Create PWM instance for vertical servo with frequency 50 Hz
servo_horizontal.start(0)  # Start PWM for horizontal servo with 0 duty cycle (neutral position)
servo_vertical.start(0)    # Start PWM for vertical servo with 0 duty cycle (neutral position)

# Initialize variables for smoothing servo movement
prev_servo_angle_horizontal = 90
prev_servo_angle_vertical = 90
smooth_factor = 0.8  # Adjust this factor for smoothing servo movement

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def draw_face_mesh(frame, face_landmarks):
    for connection in mp_face_mesh.FACEMESH_TESSELATION:
        start_idx, end_idx = connection
        cv2.line(frame, 
                 (int(face_landmarks.landmark[start_idx].x * frame.shape[1]), 
                  int(face_landmarks.landmark[start_idx].y * frame.shape[0])),
                 (int(face_landmarks.landmark[end_idx].x * frame.shape[1]), 
                  int(face_landmarks.landmark[end_idx].y * frame.shape[0])),
                 (0, 255, 0), 1)

def move_servos_to_center(x_center, y_center, smooth_factor=0.8):
    global prev_servo_angle_horizontal, prev_servo_angle_vertical
    
    # Calculate servo angles based on the center coordinates
    angle_horizontal = 90 + int((x_center - 0.5) * 180)
    angle_vertical = 90 + int((y_center - 0.5) * 180)
    
    # Smooth servo movements
    servo_angle_horizontal = smooth_factor * prev_servo_angle_horizontal + (1 - smooth_factor) * angle_horizontal
    servo_angle_vertical = smooth_factor * prev_servo_angle_vertical + (1 - smooth_factor) * angle_vertical
    
    # Limit servo angles within a safe range
    servo_angle_horizontal = max(0, min(servo_angle_horizontal, 180))
    servo_angle_vertical = max(0, min(servo_angle_vertical, 180))
    
    # Move the servos to the calculated angles
    servo_horizontal.ChangeDutyCycle(2 + (servo_angle_horizontal / 18))
    servo_vertical.ChangeDutyCycle(2 + (servo_angle_vertical / 18))
    
    # Update previous servo angles
    prev_servo_angle_horizontal = servo_angle_horizontal
    prev_servo_angle_vertical = servo_angle_vertical

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for face mesh detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect face mesh
    results_face_mesh = face_mesh_module.process(frame_rgb)

    # Draw face mesh on the frame
    if results_face_mesh.multi_face_landmarks:
        for face_landmarks in results_face_mesh.multi_face_landmarks:
            draw_face_mesh(frame, face_landmarks)
            
            # Get bounding box around face mesh
            x_min = min(face_landmarks.landmark, key=lambda point: point.x).x * frame.shape[1]
            x_max = max(face_landmarks.landmark, key=lambda point: point.x).x * frame.shape[1]
            y_min = min(face_landmarks.landmark, key=lambda point: point.y).y * frame.shape[0]
            y_max = max(face_landmarks.landmark, key=lambda point: point.y).y * frame.shape[0]
            
            # Calculate center coordinates of the bounding box
            x_center = (x_min + x_max) / 2 / frame.shape[1]
            y_center = (y_min + y_max) / 2 / frame.shape[0]
            
            # Move servos to center the face
            move_servos_to_center(x_center, y_center)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

# Clean up GPIO
servo_horizontal.stop()
servo_vertical.stop()
GPIO.cleanup()

