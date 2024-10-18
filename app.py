from flask import Flask, render_template, Response, jsonify
import cv2
import yoga
import yoga_angle
import mediapipe as mp

app = Flask(__name__)
camera = None
pose_detection_active = False
feedback = "No feedback yet."
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detection_1')
def detection_1():
    return render_template('detection_1.html')


@app.route('/detection_2')
def detection_2():
    return render_template('detection_2.html')

def init_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # Initialize only if not already open

def generate_frames():
    global pose_detection_active, feedback

    while True:
        if pose_detection_active:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame = cv2.flip(frame, 1)
                output_image, landmarks = yoga_angle.detectPose(frame, pose)  # Get landmarks and output image
                feedback = yoga_angle.classifyPose(landmarks)  # Classify the pose

                # Convert the frame to JPEG and yield it
                _, buffer = cv2.imencode('.jpg', output_image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_yoga_frames():
    global pose_detection_active, feedback

    while True:
        if pose_detection_active:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame = cv2.flip(frame, 1)
                output_image = frame

                feedback = yoga.classifyPose(output_image)

                _, buffer = cv2.imencode('.jpg', output_image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/start_pose_detection')
def start_pose_detection():
    global pose_detection_active
    pose_detection_active = True
    init_camera()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_pose_detection', methods=['POST'])
def stop_pose_detection():
    global camera
    global feedback
    global pose_detection_active
    pose_detection_active = False
    if camera is not None:
        camera.release()  # Explicitly release the camera
        camera = None
    feedback = 'Pose detection stopped'
    return jsonify({'message': 'Pose detection stopped'})

@app.route('/start_yoga_pose_detection')
def start_yoga_pose_detection():
    global pose_detection_active
    pose_detection_active = True
    init_camera()
    return Response(generate_yoga_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_yoga_pose_detection', methods=['POST'])
def stop_yoga_pose_detection():
    global camera
    global feedback
    global pose_detection_active
    pose_detection_active = False
    if camera is not None:
        camera.release()  # Explicitly release the camera
        camera = None
    feedback = 'Pose detection stopped'
    return jsonify({'message': 'Pose detection stopped'})

@app.route('/feedback')
def get_feedback():
    global feedback
    return jsonify({'feedback': feedback})

if __name__ == '__main__':
    app.run(debug=True)
