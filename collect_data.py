import os
import time
import csv
import cv2
import mediapipe as mp
import numpy as np

from utils.distance import extract_distance_features
from utils.config import dataset_name, cameras

# --> Konfigurasi data yg akan dicollect
label: str = "J"
""  # --> Ganti sesuai huruf yang ingin direkam
dataset_path: str = dataset_name  # --> Path dataset untuk mode statis (CSV)
# --> Jumlah data per huruf (atau jumlah sequence untuk dynamic)
samples: int = 100

# Mode perekaman: 'static' = frame tunggal (sekarang), 'dynamic' = sequence bergerak
mode = 'dynamic'  # pilih 'static' atau 'dynamic'

# Konfigurasi untuk mode dynamic
# jumlah frame per sequence (contoh: 30 frames ~ 1 detik @30fps)
sequence_length = 100
# jumlah frame tanpa deteksi tangan yang masih diijinkan per sequence
max_missing_frames = 100
sequences_dir = os.path.join('data', 'sequences')

# --> Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --> Webcam
cap = cv2.VideoCapture(0)


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


if mode == 'static':
    # Perilaku lama: simpan features per frame ke CSV
    data = []
    saved = 0

    print(f"Mulai merekam data untuk label '{label.upper()}' dalam 3 detik...")
    time.sleep(3)
    print("Merekam...")

    while saved < samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        image = frame.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                features = extract_distance_features(hand_landmarks.landmark)
                features.append(label.upper())
                data.append(features)
                saved += 1
                print(f"Tersimpan: {saved}/{samples}")

        cv2.imshow("Collecting Data", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --> Simpan ke CSV
    if len(data) == 0:
        print(
            f"⚠️ Tidak ada data yang direkam untuk label '{label.upper()}'. Tidak ada file yang disimpan.")
    else:
        if not os.path.exists(os.path.dirname(dataset_path)):
            os.makedirs(os.path.dirname(dataset_path))

        header = [f"d{i}" for i in range(len(data[0]) - 1)] + ["label"]
        with open(dataset_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerows(data)

        print(
            f"✅ Data untuk label '{label.upper()}' berhasil disimpan di {dataset_path}")

else:
    # Mode dynamic: simpan sequence per sample (.npy) dan catat label di sequences/labels.csv
    ensure_dir(sequences_dir)
    labels_csv = os.path.join(sequences_dir, 'labels.csv')
    # ensure labels file exists
    if not os.path.exists(labels_csv):
        with open(labels_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])

    saved = 0
    sample_idx = len([name for name in os.listdir(
        sequences_dir) if name.endswith('.npy')])

    print(
        f"Mode DYNAMIC: Akan merekam {samples} sequence untuk label '{label.upper()}'.")
    print("Instruksi: Tekan SPASI untuk mulai merekam 1 sequence, tekan 'q' untuk keluar.")

    try:
        while saved < samples:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            display = frame.copy()
            cv2.putText(display, f"Ready for sequence {saved+1}/{samples} - Press SPACE to record", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Collecting Sequences', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('Diberhentikan oleh user.')
                break
            if key == ord(' '):
                # Mulai merekam satu sequence
                seq_features = []
                missing = 0
                frames_recorded = 0
                print(f"Merekam sequence {saved+1}...")
                while frames_recorded < sequence_length:
                    ret2, frame2 = cap.read()
                    if not ret2 or frame2 is None:
                        cv2.waitKey(5)
                        continue

                    image_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    result = hands.process(image_rgb)
                    if result.multi_hand_landmarks:
                        for hand_landmarks in result.multi_hand_landmarks:
                            feats = extract_distance_features(
                                hand_landmarks.landmark)
                            seq_features.append(feats)
                            frames_recorded += 1
                            missing = 0
                            mp_drawing.draw_landmarks(
                                frame2, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            break
                    else:
                        # tidak terdeteksi tangan di frame ini
                        missing += 1
                        # tambahkan small pause
                        cv2.putText(frame2, 'No hand detected', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.waitKey(1)

                    cv2.putText(frame2, f"Recording: {frames_recorded}/{sequence_length}",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.imshow('Collecting Sequences', frame2)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

                    # Jika terlalu banyak missing frames, abort sequence
                    if missing > max_missing_frames:
                        print(
                            'Sequence dibatalkan karena terlalu banyak frame tanpa deteksi tangan.')
                        break

                # Selesai merekam sequence: cek panjang
                if len(seq_features) == sequence_length:
                    # shape (sequence_length, n_features)
                    arr = np.array(seq_features, dtype=np.float32)
                    sample_idx += 1
                    fname = f"{label.upper()}_{sample_idx:04d}.npy"
                    fpath = os.path.join(sequences_dir, fname)
                    np.save(fpath, arr)
                    # catat label
                    with open(labels_csv, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([fname, label.upper()])
                    saved += 1
                    print(f"Tersimpan sequence {saved}/{samples} -> {fpath}")
                else:
                    print(
                        'Sequence tidak disimpan (panjang tidak mencukupi). Coba lagi.')

    except KeyboardInterrupt:
        print('Diberhentikan dengan KeyboardInterrupt.')

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(
        f"Selesai. Total sequence tersimpan untuk label '{label.upper()}': {saved}")
