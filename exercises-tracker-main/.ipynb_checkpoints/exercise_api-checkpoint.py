from flask import Flask, Response
import cv2 as cv
import exercise

app = Flask(__name__)

def generate_frames():
    cap = cv.VideoCapture(0)  # Change the index to the appropriate video source (e.g., 0 for webcam)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame using the exercise module
        # Replace this with your own logic or function call
        processed_frame = exercise.process_frame(frame)

        # Convert the processed frame to JPEG format
        ret, buffer = cv.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/api/video_feed', methods=['GET'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/exercise', methods=['GET'])
def get_exercise():
    response = exercise.run_exercise()
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
