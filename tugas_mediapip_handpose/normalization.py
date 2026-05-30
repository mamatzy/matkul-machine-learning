# Separating Left and Right Hands
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,       # Enables real-time continuous tracking
    max_num_hands=2,               # Detect up to two hands simultaneously
    model_complexity=1,            # 0 = fast, 1 = balanced, 2 = high accuracy
    min_detection_confidence=0.5,  # Detection threshold
    min_tracking_confidence=0.5    # Tracking confidence threshold
)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR (OpenCV) to RGB (MediaPipe format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Create a black canvas with the same size as the input frame
    black_canvas = np.zeros_like(frame)

    # Canvas placeholders for left and right cropped hands (resized to 128x128)
    left_hand_crop  = None
    right_hand_crop = None

    if results.multi_hand_landmarks:
        for handedness, hand_landmarks in zip(results.multi_handedness,
                                              results.multi_hand_landmarks):

            # Draw landmarks on the original frame
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Draw the same landmarks on the black background
            mp_draw.draw_landmarks(
                black_canvas,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ----------------- HAND CROPPING PROCESS -----------------
            label = handedness.classification[0].label  # 'Left' or 'Right'

            h, w, c = frame.shape
            xs = [lm.x * w for lm in hand_landmarks.landmark]
            ys = [lm.y * h for lm in hand_landmarks.landmark]

            min_x, max_x = int(min(xs)), int(max(xs))
            min_y, max_y = int(min(ys)), int(max(ys))

            # Add margin around bounding box
            margin = 20
            min_x = max(min_x - margin, 0)
            min_y = max(min_y - margin, 0)
            max_x = min(max_x + margin, w - 1)
            max_y = min(max_y + margin, h - 1)

            box_w = max_x - min_x
            box_h = max_y - min_y
            side  = max(box_w, box_h)

            cx = (min_x + max_x) // 2
            cy = (min_y + max_y) // 2

            start_x = max(cx - side // 2, 0)
            start_y = max(cy - side // 2, 0)
            end_x   = min(start_x + side, w)
            end_y   = min(start_y + side, h)

            # Adjust if bounding box touches image border
            side = min(end_x - start_x, end_y - start_y)
            end_x = start_x + side
            end_y = start_y + side

            # Crop the hand region from the black canvas
            hand_region = black_canvas[start_y:end_y, start_x:end_x]

            hand_h, hand_w = hand_region.shape[:2]
            square_side = max(hand_h, hand_w)
            hand_canvas = np.zeros((square_side, square_side, 3), dtype=np.uint8)

            # Center the cropped hand inside a square background
            y_off = (square_side - hand_h) // 2
            x_off = (square_side - hand_w) // 2
            hand_canvas[y_off:y_off + hand_h, x_off:x_off + hand_w] = hand_region

            # Resize to 128×128
            resized_hand = cv2.resize(hand_canvas, (128, 128), interpolation=cv2.INTER_AREA)

            if label == "Left":
                left_hand_crop = resized_hand
            elif label == "Right":
                right_hand_crop = resized_hand
            # ----------------- END CROP PROCESS -----------------

    # Display processed windows
    cv2.imshow("Original Video with Landmarks", frame)
    cv2.imshow("Landmarks on Black Background", black_canvas)

    if left_hand_crop is not None:
        cv2.imshow("Left Hand Crop (128x128)", left_hand_crop)
    if right_hand_crop is not None:
        cv2.imshow("Right Hand Crop (128x128)", right_hand_crop)

    # -------------- COMBINE LEFT + RIGHT INTO 256x256 CANVAS --------------
    combined_256 = None

    if (left_hand_crop is not None) or (right_hand_crop is not None):
        # Base canvas: 128×256
        combined_128x256 = np.zeros((128, 256, 3), dtype=np.uint8)

        # Insert left and right hands if available
        if left_hand_crop is not None:
            combined_128x256[0:128, 0:128] = left_hand_crop
        if right_hand_crop is not None:
            combined_128x256[0:128, 128:256] = right_hand_crop

        # Final centered canvas: 256×256
        combined_256 = np.zeros((256, 256, 3), dtype=np.uint8)
        y_start = (256 - 128) // 2  # Center vertically
        combined_256[y_start:y_start + 128, 0:256] = combined_128x256

        cv2.imshow("Combined Hands 256x256", combined_256)
    # -----------------------------------------------------------------------

    # Close program with ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
