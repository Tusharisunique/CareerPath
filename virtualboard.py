import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Create a black canvas for drawing
canvas = None
drawing_color = (255, 255, 255)  # White for drawing
eraser_thickness = 50  # Thickness of the eraser

# Function to detect open fingers
def count_fingers(hand_landmarks):
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    dips = [mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    open_fingers = []
    for tip, dip in zip(tips, dips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:  # Finger is open if TIP is above DIP
            open_fingers.append(tip)
    return open_fingers

# Open the webcam
cap = cv2.VideoCapture(0)

# Initialize variables
previous_position = None
mode = "surf"  # Default to "surf" mode
is_drawing = False  # To prevent drawing when switching between modes

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a selfie-view
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Initialize the canvas if not already done
    if canvas is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert the frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands in the frame
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip position
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x, index_y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

            # Count open fingers
            open_fingers = count_fingers(hand_landmarks)

            if len(open_fingers) == 1:  # Only the index finger is open
                mode = "draw"
                is_drawing = True
                if previous_position is not None:
                    cv2.line(canvas, previous_position, (index_x, index_y), drawing_color, 5)  # Draw line
                previous_position = (index_x, index_y)

            elif len(open_fingers) == 2:  # Both index and middle fingers are open
                mode = "erase"
                is_drawing = False
                previous_position = None  # Reset the previous position to prevent unwanted lines
                cv2.circle(canvas, (index_x, index_y), eraser_thickness, (0, 0, 0), -1)  # Erase

            else:  # No fingers are open (fist or neutral)
                mode = "surf"
                is_drawing = False
                previous_position = None  # Reset the previous position

    else:
        # If no hand is detected, reset to surf mode
        mode = "surf"
        is_drawing = False
        previous_position = None

    # Combine the canvas with the frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the current mode
    cv2.putText(combined, f"Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Virtual Whiteboard", combined)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
