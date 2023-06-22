from flask import Flask, jsonify, request
import numpy as np
import cv2
import subprocess

# Create the Flask app
app = Flask(__name__)

# Define the API endpoint for receiving video frames
@app.route('/video_frame', methods=['GET'])
def video_frame():
    # Get the video frame data from the request
    frame_data = request.data

    # Convert the frame data to a numpy array
    frame = np.frombuffer(frame_data, dtype=np.uint8)

    # Reshape the frame to its original dimensions
    # This assumes that the frame is in RGB format
    frame = frame.reshape((480, 640, 3))

    # Run the Python script using subprocess
    script_path = 'exercise.py'  # Replace with the actual path to your script
    result = subprocess.run(['python', script_path], capture_output=True, text=True)

    # Process the result and return it as a JSON response
    # Assuming your script returns a JSON object
    pose_data = result.stdout  # Modify this according to the output of your script

    return jsonify({'pose_data': pose_data})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
