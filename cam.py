import cv2, pickle, numpy as np, mediapipe as mp, tensorflow as tf
from collections import deque

from utils.distance import extract_distance_features
from utils.config import scaler_name, binarizer_name, model_name, cameras

#--> Load model dan preprocessing
model = tf.keras.models.load_model(model_name)
with open(scaler_name, "rb") as f:
    scaler = pickle.load(f)
with open(binarizer_name, "rb") as f:
    lb = pickle.load(f)

# detect input shape: static (None, n_features) or dynamic (None, seq_len, n_features)
input_shape = getattr(model, 'input_shape', None)
is_dynamic = False
seq_len = None
n_features_model = None
if input_shape is not None:
    try:
        # Keras may return tuple or list
        if hasattr(input_shape, '__len__') and len(input_shape) == 3:
            is_dynamic = True
            seq_len = int(input_shape[1])
            n_features_model = int(input_shape[2])
        elif hasattr(input_shape, '__len__') and len(input_shape) == 2:
            is_dynamic = False
            n_features_model = int(input_shape[1])
    except Exception:
        pass

print(f"Model input_shape={input_shape}, is_dynamic={is_dynamic}, seq_len={seq_len}, n_features={n_features_model}")

#--> Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

#--> Webcam (pakai cameras dari config jika tersedia)
cap_index = cameras if (cameras is not None) else 0
cap = cv2.VideoCapture(0)

# rolling buffer untuk continuous dynamic inference
rolling_enabled = True
pred_history = deque(maxlen=5)  # smoothing of last predictions

if is_dynamic:
    buffer = deque(maxlen=seq_len)
else:
    buffer = deque(maxlen=1)

print("Controls: 'q' quit, SPACE record (one-shot dynamic), 'r' toggle rolling continuous prediction")

def predict_static(features):
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled, verbose=0)
    idx = int(np.argmax(pred))
    return lb.classes_[idx], float(pred[0][idx])

def predict_sequence(seq_array):
    # seq_array shape (seq_len, n_features)
    Xs = np.asarray(seq_array, dtype=np.float32)
    Xs_flat = Xs.reshape(-1, Xs.shape[-1])
    Xs_flat_scaled = scaler.transform(Xs_flat)
    Xs_scaled = Xs_flat_scaled.reshape(1, Xs.shape[0], Xs.shape[1])
    pred = model.predict(Xs_scaled, verbose=0)
    idx = int(np.argmax(pred))
    return lb.classes_[idx], float(pred[0][idx])

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = frame.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        features = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                feats = extract_distance_features(hand_landmarks.landmark)
                features = np.array(feats, dtype=np.float32)
                break

        # Static model inference
        if not is_dynamic:
            if features is not None:
                if n_features_model is not None and features.shape[0] != n_features_model:
                    cv2.putText(image, f"Feat mismatch {features.shape[0]}!={n_features_model}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                else:
                    label, conf = predict_static(features)
                    cv2.putText(image, f"{label} ({conf*100:.1f}%)", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)

        # Dynamic model handling
        else:
            # Update buffer continuously if a hand is detected
            if features is not None:
                if n_features_model is not None and features.shape[0] != n_features_model:
                    cv2.putText(image, f"Feat mismatch {features.shape[0]}!={n_features_model}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
                else:
                    buffer.append(features)

            # rolling continuous prediction
            if rolling_enabled and len(buffer) == seq_len:
                label, conf = predict_sequence(list(buffer))
                pred_history.append((label, conf))
                # smoothing: most common label in pred_history
                labels = [p[0] for p in pred_history]
                best = max(set(labels), key=labels.count)
                # take last confidence for best label
                last_conf = next((p[1] for p in reversed(pred_history) if p[0]==best), 0.0)
                cv2.putText(image, f"{best} ({last_conf*100:.1f}%)", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255),2)
            else:
                cv2.putText(image, f"Dynamic mode - buffer {len(buffer)}/{seq_len}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

        cv2.imshow('Hand Sign Recognition', image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            rolling_enabled = not rolling_enabled
            print('Rolling prediction:', rolling_enabled)
        if is_dynamic and key == ord(' '):
            # manual one-shot record of current buffer (if full) or record next seq_len frames
            if len(buffer) == seq_len:
                label, conf = predict_sequence(list(buffer))
                print(f'One-shot prediction (buffer): {label} ({conf*100:.1f}%)')
            else:
                print('Recording one-shot sequence...')
                seq = []
                missing = 0
                while len(seq) < seq_len:
                    ret2, f2 = cap.read()
                    if not ret2:
                        continue
                    f_rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
                    r2 = hands.process(f_rgb)
                    feat2 = None
                    if r2.multi_hand_landmarks:
                        for hl in r2.multi_hand_landmarks:
                            feat2 = np.array(extract_distance_features(hl.landmark), dtype=np.float32)
                            break
                    if feat2 is None:
                        missing += 1
                        if missing > 10:
                            print('Too many missing frames, aborting one-shot.')
                            break
                        continue
                    if feat2.shape[0] != n_features_model:
                        print('Feature dim mismatch during one-shot, aborting')
                        break
                    seq.append(feat2)
                if len(seq) == seq_len:
                    label, conf = predict_sequence(seq)
                    print(f'One-shot prediction: {label} ({conf*100:.1f}%)')

except KeyboardInterrupt:
    print('Stopped by user')

finally:
    cap.release()
    cv2.destroyAllWindows()