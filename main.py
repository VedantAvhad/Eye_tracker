import cv2
from eye_tracker import track_pupil  # Correct function name



def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = track_pupil(frame)  # Apply pupil tracking
        cv2.imshow("Pupil Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
