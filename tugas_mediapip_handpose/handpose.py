import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands                     # Loads the MediaPipe Hands module
mp_draw = mp.solutions.drawing_utils             # Provides drawing functions for landmarks
hands = mp_hands.Hands()                         # Creates a Hands detector with default settings

# Initialize webcam capture (device index 0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the image from BGR (OpenCV default) to RGB (MediaPipe requirement)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect hand landmarks
    results = hands.process(rgb)

    # Check if any hand landmark is detected
    if results.multi_hand_landmarks:
        # Loop through each detected hand and draw its landmarks
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    # Display the resulting output frame
    cv2.imshow("MediaPipe Hands", frame)

    # Exit condition: press the 'ESC' key (key code 27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release camera resources and close display windows
cap.release()
cv2.destroyAllWindows()
