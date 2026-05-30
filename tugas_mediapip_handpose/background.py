import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,      # Enables real-time tracking mode
    max_num_hands=2,              # Detect up to two hands simultaneously
    model_complexity=1,           # Balances accuracy and speed
    min_detection_confidence=0.5, # Minimum confidence threshold for initial detection
    min_tracking_confidence=0.5   # Confidence required to keep tracking after detection
)

# Open webcam stream
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR (OpenCV) to RGB format required by MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Prepare black canvas with the same dimensions as the frame
    black_canvas = np.zeros_like(frame)

    # If hand landmarks are detected, draw them on both images
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw landmarks and connections on the original camera frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw the same landmarks on a fully black background
            mp_draw.draw_landmarks(black_canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the output windows:
    # 1. Camera frame with detected hand landmarks
    # 2. Isolated hand landmarks over a black background
    cv2.imshow("Original Video with Landmarks", frame)
    cv2.imshow("Landmarks on Black Background", black_canvas)

    # Exit the loop when the ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release webcam and close all display windows
cap.release()
cv2.destroyAllWindows()
hands.close()
