from poseModule import PoseDetector, draw_landmarks
import cv2 as cv
import numpy as np
from mediapipe.python.solutions import pose as mp_pose


def calc_angle(a, b, c):
    """
    Calculate the angle between three points
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def draw(frame, a, b):
    """
    Draw a line and circle between two points
    """
    x1, y1 = int(a[1]), int(a[2])
    x2, y2 = int(b[1]), int(b[2])

    cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv.circle(frame, (x1, y1), 5, (0, 0, 255), cv.FILLED)
    cv.circle(frame, (x2, y2), 5, (0, 0, 255), cv.FILLED)


def main():
    cap = cv.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    writer = cv.VideoWriter('pose_estimation.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    pose_detector = PoseDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_detector.detect_pose(frame)

        if results.pose_landmarks:
            draw_landmarks(frame, results.pose_landmarks)

            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angle = calc_angle(left_shoulder, left_elbow, left_wrist)
            cv.putText(frame, f'{int(angle)}', (20, 100), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)

            draw(frame, left_shoulder, left_elbow)
            draw(frame, left_elbow, left_wrist)

        cv.imshow('Pose Estimation', frame)
        writer.write(frame)

        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    writer.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
