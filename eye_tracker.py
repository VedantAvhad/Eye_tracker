import cv2
import numpy as np
import dlib

# Load face and eye detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

eye_movement_count = 0  # Stores eyeball movement count
looking_outside = False  # Flag to track when user is looking outside

def detect_pupil(eye_points, frame):
    """Detects the pupil by finding the darkest region in the eye."""
    x_min, y_min = np.min(eye_points, axis=0)
    x_max, y_max = np.max(eye_points, axis=0)

    eye_region = frame[y_min:y_max, x_min:x_max]
    eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY_INV)

    moments = cv2.moments(thresh)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"]) + x_min
        cy = int(moments["m01"] / moments["m00"]) + y_min
        return cx, cy, (x_min, y_min, x_max, y_max)
    return None, None, (x_min, y_min, x_max, y_max)

def check_looking_direction(pupil_x, eye_bounds):
    """Detects if the pupil is moving left or right inside the eye."""
    x_min, _, x_max, _ = eye_bounds
    eye_width = x_max - x_min
    left_threshold = x_min + eye_width * 0.3
    right_threshold = x_min + eye_width * 0.7

    if pupil_x is None:
        return None  # If eye is closed, return None

    if pupil_x < left_threshold:
        return "Looking Left"
    elif pupil_x > right_threshold:
        return "Looking Right"
    return "Looking Center"

def track_pupil(frame):
    """Tracks eyes and detects if the person looks away or moves eyeballs left/right."""
    global eye_movement_count, looking_outside

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        pupil_l_x, pupil_l_y, left_eye_bounds = detect_pupil(left_eye_points, frame)
        pupil_r_x, pupil_r_y, right_eye_bounds = detect_pupil(right_eye_points, frame)

        # Draw eyes and pupil
        cv2.polylines(frame, [left_eye_points], True, (255, 255, 0), 1)
        cv2.polylines(frame, [right_eye_points], True, (255, 255, 0), 1)

        if pupil_l_x and pupil_l_y:
            cv2.circle(frame, (pupil_l_x, pupil_l_y), 3, (0, 0, 255), -1)

        if pupil_r_x and pupil_r_y:
            cv2.circle(frame, (pupil_r_x, pupil_r_y), 3, (0, 0, 255), -1)

        # **Eyeball Movement Detection**
        left_eye_direction = check_looking_direction(pupil_l_x, left_eye_bounds)
        right_eye_direction = check_looking_direction(pupil_r_x, right_eye_bounds)

        if left_eye_direction or right_eye_direction:
            print(f"Left Eye: {left_eye_direction} | Right Eye: {right_eye_direction}")

        # **Cheating Detection - Looking Outside the Screen**
        if left_eye_direction == "Looking Left" and right_eye_direction == "Looking Left":
            looking_outside = True
            cv2.putText(frame, "Looking Outside!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if left_eye_direction == "Looking Right" and right_eye_direction == "Looking Right":
            looking_outside = True
            cv2.putText(frame, "Looking Outside!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # **Count Eyeball Movements After Looking Outside**
        if looking_outside and (left_eye_direction in ["Looking Left", "Looking Right"] or 
                                right_eye_direction in ["Looking Left", "Looking Right"]):
            eye_movement_count += 1
            print(f"Eyeball Moved {eye_movement_count} Times After Looking Outside!")

    return frame
