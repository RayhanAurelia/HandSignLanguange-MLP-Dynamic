from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import json

from utils.distance import extract_distance_features
from utils.config import scaler_name, binarizer_name, model_name
from utils.eye import get_eye_ratio

app = Flask(__name__)

# Load model and preprocessors
model = tf.keras.models.load_model(model_name)
with open(scaler_name, "rb") as f:
    scaler = pickle.load(f)
with open(binarizer_name, "rb") as f:
    lb = pickle.load(f)

# Detect model type
input_shape = getattr(model, 'input_shape', None)
is_dynamic = False
seq_len = None
n_features_model = None

if input_shape is not None:
    try:
        if hasattr(input_shape, '__len__') and len(input_shape) == 3:
            is_dynamic = True
            seq_len = int(input_shape[1])
            n_features_model = int(input_shape[2])
        elif hasattr(input_shape, '__len__') and len(input_shape) == 2:
            is_dynamic = False
            n_features_model = int(input_shape[1])
    except Exception:
        pass

print(f"Model type: {'Dynamic (LSTM)' if is_dynamic else 'Static (MLP)'}")
print(f"Input shape: {input_shape}")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Global state
current_prediction = {"label": "No hand detected", "confidence": 0.0}
pred_history = deque(maxlen=5)
final_text = ""
last_saved_label = ""
blink_cooldown = 0
wave_cooldown = 0
prev_wrist_x = None

# Wave detection threshold
WAVE_THRESHOLD = 80

if is_dynamic:
    buffer = deque(maxlen=seq_len)
else:
    buffer = deque(maxlen=1)


def predict_static(features):
    """Predict from single frame features"""
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled, verbose=0)
    idx = int(np.argmax(pred))
    return lb.classes_[idx], float(pred[0][idx])


def predict_sequence(seq_array):
    """Predict from sequence of features"""
    Xs = np.asarray(seq_array, dtype=np.float32)
    Xs_flat = Xs.reshape(-1, Xs.shape[-1])
    Xs_flat_scaled = scaler.transform(Xs_flat)
    Xs_scaled = Xs_flat_scaled.reshape(1, Xs.shape[0], Xs.shape[1])
    pred = model.predict(Xs_scaled, verbose=0)
    idx = int(np.argmax(pred))
    return lb.classes_[idx], float(pred[0][idx])


def generate_frames():
    """Generate frames for video streaming"""
    global current_prediction, final_text, last_saved_label, blink_cooldown, wave_cooldown, prev_wrist_x

    cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            image = frame.copy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # ==== Detect Blink ==== #
            face_result = face_mesh.process(image_rgb)
            blink_now = False

            if face_result.multi_face_landmarks:
                for face_landmarks in face_result.multi_face_landmarks:
                    landmarks = [(int(lm.x * w), int(lm.y * h))
                                 for lm in face_landmarks.landmark]
                    left_eye_ratio = get_eye_ratio(
                        landmarks, [33, 159, 160, 133, 144, 145])
                    right_eye_ratio = get_eye_ratio(
                        landmarks, [362, 386, 387, 263, 373, 374])
                    avg_ratio = (left_eye_ratio + right_eye_ratio) / 2

                    if avg_ratio < 0.20 and blink_cooldown == 0:
                        blink_now = True
                        blink_cooldown = 10

            if blink_cooldown > 0:
                blink_cooldown -= 1

            # ==== Hand Detection ==== #
            results = hands.process(image_rgb)

            features = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # ==== Detect Wave (Undo) ==== #
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    if prev_wrist_x is not None and wave_cooldown == 0:
                        if abs(wrist_x - prev_wrist_x) > WAVE_THRESHOLD:
                            if len(final_text) > 0:
                                final_text = final_text[:-1]
                                wave_cooldown = 15
                    prev_wrist_x = wrist_x

                    if wave_cooldown > 0:
                        wave_cooldown -= 1

                    # Extract features
                    feats = extract_distance_features(hand_landmarks.landmark)
                    features = np.array(feats, dtype=np.float32)
                    break

            # Make prediction
            if not is_dynamic:
                # Static model
                if features is not None:
                    if n_features_model is not None and features.shape[0] != n_features_model:
                        current_prediction = {
                            "label": "Feature mismatch",
                            "confidence": 0.0
                        }
                    else:
                        label, conf = predict_static(features)
                        pred_history.append((label, conf))

                        # Smooth predictions
                        if len(pred_history) >= 3:
                            labels = [p[0] for p in pred_history]
                            most_common = max(set(labels), key=labels.count)
                            avg_conf = np.mean(
                                [p[1] for p in pred_history if p[0] == most_common])
                            current_prediction = {
                                "label": most_common,
                                "confidence": float(avg_conf)
                            }

                            # Save on blink if confidence > 50%
                            if blink_now and avg_conf > 0.5:
                                final_text += most_common
                                last_saved_label = most_common
                                blink_now = False
                        else:
                            current_prediction = {
                                "label": label,
                                "confidence": float(conf)
                            }

                            # Save on blink if confidence > 50%
                            if blink_now and conf > 0.5:
                                final_text += label
                                last_saved_label = label
                                blink_now = False
                else:
                    current_prediction = {
                        "label": "No hand detected",
                        "confidence": 0.0
                    }
            else:
                # Dynamic model
                if features is not None:
                    buffer.append(features)

                    if len(buffer) == seq_len:
                        label, conf = predict_sequence(list(buffer))
                        current_prediction = {
                            "label": label,
                            "confidence": float(conf)
                        }

                        # Save on blink if confidence > 50%
                        if blink_now and conf > 0.5:
                            final_text += label
                            last_saved_label = label
                            blink_now = False
                    else:
                        current_prediction = {
                            "label": f"Collecting frames ({len(buffer)}/{seq_len})",
                            "confidence": 0.0
                        }
                else:
                    current_prediction = {
                        "label": "No hand detected",
                        "confidence": 0.0
                    }

            # Draw prediction on frame
            text = f"{current_prediction['label']}"
            conf_text = f"{current_prediction['confidence']*100:.1f}%"

            cv2.putText(image, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(image, conf_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw saved text
            cv2.putText(image, f"Saved: {final_text}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            # Encode frame
            ret, buffer_frame = cv2.imencode('.jpg', image)
            frame_bytes = buffer_frame.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        cap.release()


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html',
                           model_type='Dynamic (LSTM)' if is_dynamic else 'Static (MLP)',
                           labels=lb.classes_.tolist())


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/prediction')
def get_prediction():
    """Get current prediction as JSON"""
    return jsonify({
        **current_prediction,
        "saved_text": final_text
    })


@app.route('/clear_text', methods=['POST'])
def clear_text():
    """Clear saved text"""
    global final_text
    final_text = ""
    return jsonify({"status": "success", "saved_text": ""})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
