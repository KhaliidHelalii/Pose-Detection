import cv2 as cv
import mediapipe as mp
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import numpy as np
import math

app = Flask(__name__)
CORS(app)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

count = 0
position = 'up'
startAngle, endAngle = None, None
exercise = 'pull up'

def draw(frame, point1, point2, valid=True):
    x1, y1 = int(point1[1]), int(point1[2])
    x2, y2 = int(point2[1]), int(point2[2])

    color = (0, 255, 0) if valid else (0, 0, 255)

    cv.line(frame, (x1, y1), (x2, y2), color, 3)
    cv.circle(frame, (x2, y2), 5, color, cv.FILLED)
    cv.circle(frame, (x1, y1), 5, color, cv.FILLED)
    cv.circle(frame, (x1, y1), 10, (230, 230, 230), 5)
    cv.circle(frame, (x2, y2), 10, (230, 230, 230), 5)

def push_pull(frame, points, pull=False):
    global count, position, startAngle, endAngle

    if pull:
        start = 40
        end = 165
    else:
        start = 183
        end = 300

    startAngle, endAngle = start, end

    draw(frame, points[11], points[13])
    draw(frame, points[13], points[15])
    draw(frame, points[12], points[14])
    draw(frame, points[14], points[16])

    if points[12][2] and points[11][2] >= points[14][2] and points[13][2]:
        position = 'down'

    if (points[12][2] and points[11][2] <= points[14][2] and points[13][2]) and position == 'down':
        count += 1
        position = 'up'
        cv.putText(frame, "Push-up counted!", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

def generate_frames(exercise_type):
    global exercise
    exercise = exercise_type

    cap = cv.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    writer = cv.VideoWriter('pose_estimation.mp4', cv.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                landmarks = results.pose_landmarks.landmark
                points = [[index, landmark.x, landmark.y, landmark.visibility] for index, landmark in enumerate(landmarks)]

                if exercise.lower() == 'push-ups':
                    push_pull(frame, points, pull=False)
                elif exercise.lower() == 'pull-ups':
                    push_pull(frame, points, pull=True)
                else:
                    cv.putText(frame, "Invalid exercise type", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            cv.putText(frame, f"Push-ups: {count}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            writer.write(frame_rgb)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    writer.release()
    cap.release()
    cv.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    global exercise
    exercise = request.args.get('exercise')
    return Response(generate_frames(exercise), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
