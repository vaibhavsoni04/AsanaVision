import tensorflow as tf
import cv2
import numpy as np
# Load the model from a file
model = tf.keras.models.load_model('Yoga_Pose_Classify_model.keras')

input_shape = (180, 180)

pose_labels = ['Downdog', 'Goddess', 'Plank', 'Tree', 'Warrior']

confidence_threshold = 0.3

def detectPose(image):
    output_image = image.copy()
    height, width, _ = image.shape

    return output_image

def classifyPose(frame):

    img = cv2.resize(frame, input_shape)
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)

    score = tf.nn.softmax(predictions[0])

    max_confidence_index = np.argmax(score)

    max_confidence_score = score[max_confidence_index]

    if max_confidence_score < confidence_threshold:
        predicted_pose = "Unknown Pose"
    else:
        predicted_pose = pose_labels[max_confidence_index]

    return predicted_pose