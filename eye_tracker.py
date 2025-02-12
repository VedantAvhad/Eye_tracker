import cv2
import numpy as np
import dlib

# Load face and eye detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

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

def track_pupil(frame):
    """Tracks eyes and detects if the person looks away."""
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

        # **Cheating Detection - Looking Outside the Screen**
        def check_looking_away(pupil_x, pupil_y, eye_bounds):
            """Detects if the pupil moves too far left, right, up, or down."""
            x_min, y_min, x_max, y_max = eye_bounds
            margin = 6
            if pupil_x and pupil_y:
                if pupil_x < x_min + margin or pupil_x > x_max - margin:  # Looking extreme left or right
                    return True
                if pupil_y < y_min + margin or pupil_y > y_max - margin:  # Looking extreme up or down
                    return True
            return False

        if check_looking_away(pupil_l_x, pupil_l_y, left_eye_bounds) or check_looking_away(pupil_r_x, pupil_r_y, right_eye_bounds):
            # print("⚠️ Looking Outside Detected! ⚠️")
            cv2.putText(frame, "Looking Outside!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame
