import cv2
import mediapipe as mp

# Initialize MediaPipe modules
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Configure the Hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start the webcam stream
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert image format from BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # If hands are detected, extract and display landmarks
    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

            # Draw connections and keypoints on screen
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            print(f"\nDetected Hand: {hand_index + 1}")

            # Extract normalized and pixel coordinates for each landmark
            for idx, lm in enumerate(hand_landmarks.landmark):
                height, width, _ = frame.shape
                # Extract the pixel coordinates of the landmark by multiplying the
                # normalized values (0–1 range) with the actual frame dimensions.
                x_pixel, y_pixel = int(lm.x * width), int(lm.y * height)

                print(f"Landmark {idx:02d} → "
                      f"x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:.4f} | "
                      f"Pixel=({x_pixel}, {y_pixel})")

                # Optional: draw the exact pixel point on the frame
                cv2.circle(frame, (x_pixel, y_pixel), 4, (0, 255, 0), -1)

    # Display the processed frame
    cv2.imshow("MediaPipe Hands - Landmark Extraction", frame)

    # Press ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
