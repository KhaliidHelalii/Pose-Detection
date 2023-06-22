import cv2 as cv
import mediapipe as mp
import numpy as np
import math
from flask import Flask, jsonify, request

app = Flask(__name__)

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth=True, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.detection_confidence = bool(detection_confidence)  # Convert to bool
        self.tracking_confidence = bool(tracking_confidence)  # Convert to bool
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            self.mode,
            self.complexity,
            self.smooth,
            self.detection_confidence,
            self.tracking_confidence
        )
        self.results = None

    def estimate(self, image, draw=True):
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(image_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.draw_landmarks(image, self.results.pose_landmarks)
        return image

    def draw_landmarks(self, image, landmarks):
        # Draw landmarks on the image using OpenCV
        pass

    def find_positions(self, image, draw=True):
        landmark_list = []
        if self.results.pose_landmarks:
            for id, landmark in enumerate(self.results.pose_landmarks.landmark):
                height, width, _ = image.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append([id, cx, cy])
                if draw:
                    cv.circle(image, (cx, cy), 5, (0, 255, 0), cv.FILLED)
        return landmark_list

def calc_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang

def draw(frame, point1, point2):
    x1, y1 = point1[1], point1[2]
    x2, y2 = point2[1], point2[2]
    cv.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
    cv.circle(frame, (x2, y2), 5, (255, 255, 255), cv.FILLED)
    cv.circle(frame, (x1, y1), 5, (255, 255, 255), cv.FILLED)
    cv.circle(frame, (x1, y1), 10, (230, 230, 230), 5)
    cv.circle(frame, (x2, y2), 10, (230, 230, 230), 5)

# variables
count = 0
position = 'up'
angle, startAngle, endAngle = None, None, None

def push_pull(pull=False):
    # Implementation of the push_pull function
    pass

def abdominal():
    # Implementation of the abdominal function
    pass

@app.route('/camera', methods=['GET'])
def run_camera():
    exercise = request.args.get('exercise')
    detector = PoseDetector()

    cap = cv.VideoCapture(0)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter('pose_estimation.mp4', fourcc, 30, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.estimate(frame, draw=True)
        points = detector.find_positions(frame, draw=True)
        if len(points) > 0:
            volBar = 0  # Initialize volBar with a default value
            if 'angle' not in locals():
             angle = 0
  # Initialize angle with a default value

            if exercise == 'push':
                push_pull()
                volBar = np.interp(angle, [startAngle, endAngle], [400, 150])
                volPer = np.interp(angle, [startAngle, endAngle], [0, 100])
            elif exercise == 'pull':
                push_pull(pull=True)
                volBar = np.interp(angle, [startAngle, endAngle], [150, 400])
                volPer = np.interp(angle, [startAngle, endAngle], [100, 0])
            elif exercise == 'abdominal':
                abdominal()
                volBar = np.interp(angle, [startAngle, endAngle], [150, 400])
                volPer = np.interp(angle, [startAngle, endAngle], [100, 0])

            # bar display
            # cv.rectangle(frame, (50, 150), (60, 400), (230, 230, 230), cv.FILLED)
            # cv.rectangle(frame, (50, int(volBar)), (60, 400), (30, 30, 30), cv.FILLED)
            # cv.putText(frame, str(int(volPer)) + "%", (88, int(volBar)), cv.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 2)

            # counter display
            cv.circle(frame, (60, 40), 33, (320, 320, 320), cv.FILLED)
            cv.circle(frame, (60, 40), 33, (50, 50, 50), 5)
            if count < 10:
                cv.putText(frame, str(count), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
            else:
                cv.putText(frame, str(count), (40, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        writer.write(frame)
        cv.imshow('frame', frame)
        if cv.waitKey(1) == 27:
            break

    writer.release()
    cap.release()
    cv.destroyAllWindows()

    return jsonify({'message': 'Camera is running.'})

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
