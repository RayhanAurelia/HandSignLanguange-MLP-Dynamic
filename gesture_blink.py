import cv2
import pickle
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

from utils.distance import extract_distance_features
from utils.config import scaler_name, binarizer_name, model_name, cameras
from utils.eye import get_eye_ratio

# --> Load model dan preprocessing
model = tf.keras.models.load_model(model_name)
with open(scaler_name, "rb") as f:
    scaler = pickle.load(f)
with open(binarizer_name, "rb") as f:
    lb = pickle.load(f)

# --> Detect model type (static or dynamic)
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

# --> Buffer for dynamic model
if is_dynamic:
    feature_buffer = deque(maxlen=seq_len)
else:
    feature_buffer = None

# --> Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# --> Webcam
cap = cv2.VideoCapture(0)

# --> Untuk menyimpan teks hasil
final_text = ""
last_label = ""
blink_cooldown = 0
wave_cooldown = 0
prev_wrist_x = None

# --> Ambang batas lambaian (gerakan horizontal)
WAVE_THRESHOLD = 80

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    image = frame.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ==== Deteksi Kedipan ==== #
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
                blink_cooldown = 10  # --> delay antar kedipan

    if blink_cooldown > 0:
        blink_cooldown -= 1

    # ==== Gesture Tangan + Lambaian ==== #
    hand_result = hands.process(image_rgb)
    if hand_result.multi_hand_landmarks:
        for hand_landmarks in hand_result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --> Deteksi Lambaian (undo)
            # --> landmark 0 = pergelangan tangan
            wrist_x = int(hand_landmarks.landmark[0].x * w)
            if prev_wrist_x is not None and wave_cooldown == 0:
                if abs(wrist_x - prev_wrist_x) > WAVE_THRESHOLD:
                    if len(final_text) > 0:
                        final_text = final_text[:-1]
                        wave_cooldown = 1  # --> delay agar tidak spam
            prev_wrist_x = wrist_x

            if wave_cooldown > 0:
                wave_cooldown -= 1

            # --> Deteksi Gesture Huruf
            features = extract_distance_features(hand_landmarks.landmark)
            if len(features) == 16:
                label = "No gesture"
                confidence = 0.0

                if not is_dynamic:
                    # Static model prediction
                    features_np = np.array(features).reshape(1, -1)
                    features_scaled = scaler.transform(features_np)
                    prediction = model.predict(features_scaled, verbose=0)
                    pred_index = np.argmax(prediction)
                    label = lb.classes_[pred_index]
                    confidence = prediction[0][pred_index] * 100
                else:
                    # Dynamic model prediction
                    feature_buffer.append(np.array(features, dtype=np.float32))

                    if len(feature_buffer) == seq_len:
                        # Prepare sequence
                        # shape: (seq_len, n_features)
                        seq_array = np.array(list(feature_buffer))
                        seq_flat = seq_array.reshape(-1, seq_array.shape[-1])
                        seq_flat_scaled = scaler.transform(seq_flat)
                        seq_scaled = seq_flat_scaled.reshape(
                            1, seq_array.shape[0], seq_array.shape[1])

                        # Predict
                        prediction = model.predict(seq_scaled, verbose=0)
                        pred_index = np.argmax(prediction)
                        label = lb.classes_[pred_index]
                        confidence = prediction[0][pred_index] * 100
                    else:
                        label = f"Collecting ({len(feature_buffer)}/{seq_len})"
                        confidence = 0.0

                cv2.putText(image, f"{label} ({confidence:.1f}%)", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

                if blink_now and confidence > 50:  # Only add if confidence > 50%
                    # if label != last_label:  #--> hanya jika label baru
                    final_text += label
                    last_label = label
                    blink_now = False

    # ==== Tampilkan Hasil ==== #
    cv2.putText(image, f"Output: {final_text}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Gesture + Blink + Undo", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
