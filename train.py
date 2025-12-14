import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix

from utils.config import dataset_name, scaler_name, binarizer_name, model_name

# If dynamic sequences exist, train temporal model instead of static MLP
sequences_dir = os.path.join('data', 'sequences')
sequences_labels = os.path.join(sequences_dir, 'labels.csv')


def load_sequences_and_labels(sequences_dir, labels_csv):
    df_seq = pd.read_csv(labels_csv)
    X_list = []
    y_list = []
    expected_shape = None
    skipped_by_label = {}

    for idx, row in df_seq.iterrows():
        fname = row['filename']
        label = row['label']
        fpath = os.path.join(sequences_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: sequence file not found {fpath}, skipping")
            skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
            continue

        try:
            arr = np.load(fpath)
        except Exception as e:
            print(f"Warning: error loading {fpath}: {e}, skipping")
            skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
            continue

        # Check shape consistency
        if expected_shape is None:
            expected_shape = arr.shape
            print(f"Reference shape set to: {expected_shape}")
        elif arr.shape != expected_shape:
            # Try to resize sequence to match expected length
            current_len, n_feat = arr.shape
            target_len, _ = expected_shape

            if n_feat == expected_shape[1]:  # Same number of features
                # Resample sequence to target length using linear interpolation indices
                indices = np.linspace(0, current_len - 1,
                                      target_len).astype(int)
                arr = arr[indices]
                if idx < 5:  # Only show first few messages
                    print(
                        f"Info: Resampled {fname} from {current_len} to {target_len} frames")
            else:
                print(
                    f"Warning: feature mismatch for {fname}. Expected {expected_shape}, got {arr.shape}. Skipping.")
                skipped_by_label[label] = skipped_by_label.get(label, 0) + 1
                continue

        X_list.append(arr.astype('float32'))
        y_list.append(label)

    if len(X_list) == 0:
        raise RuntimeError('No sequence data found.')

    print(f"\n✅ Loaded {len(X_list)} sequences with shape {expected_shape}")
    if skipped_by_label:
        print(f"⚠️  Skipped files by label: {skipped_by_label}")

    # Check labels coverage
    unique_labels = sorted(set(y_list))
    print(
        f"✅ Labels in sequences: {unique_labels} ({len(unique_labels)} classes)")

    X = np.stack(X_list, axis=0)
    y = np.array(y_list)
    return X, y


# Decide whether to train dynamic (sequence) model or static (CSV) model
is_dynamic = False
if os.path.exists(sequences_dir) and os.path.exists(sequences_labels):
    print('Detected dynamic sequences data. Preparing dynamic training pipeline...')
    is_dynamic = True
    X_seq, y_seq = load_sequences_and_labels(sequences_dir, sequences_labels)

    # If static CSV also exists, load and convert static rows to sequences (repeat rows)
    X_stat = None
    y_stat = None
    if os.path.exists(dataset_name):
        print('Also detected static CSV dataset. It will be converted to sequences and merged.')
        df = pd.read_csv(dataset_name)
        X_stat = df.drop('label', axis=1).values.astype('float32')
        y_stat = df['label'].values.astype(str)

    # determine shapes
    n_seq_samples, seq_len, n_features = X_seq.shape

    # if static exists, convert to sequences by repeating each feature vector seq_len times
    if X_stat is not None:
        n_stat_samples, n_features_stat = X_stat.shape
        if n_features_stat != n_features:
            raise RuntimeError(
                f"Feature dimension mismatch: sequences have {n_features} features but CSV has {n_features_stat} features")

        X_stat_seq = np.tile(X_stat[:, None, :], (1, seq_len, 1))

        # Check labels in static data
        unique_static_labels = sorted(set(y_stat))
        print(
            f"✅ Labels in static CSV: {unique_static_labels} ({len(unique_static_labels)} classes)")

        # combine
        X = np.concatenate([X_seq, X_stat_seq], axis=0)
        y = np.concatenate([y_seq, y_stat], axis=0)

        # Check combined labels
        unique_combined = sorted(set(y))
        print(
            f"✅ Combined labels: {unique_combined} ({len(unique_combined)} classes)")
    else:
        X = X_seq
        y = y_seq

    # Final check for all A-Z letters
    expected_letters = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    actual_letters = sorted(set(y))
    missing_letters = set(expected_letters) - set(actual_letters)

    if missing_letters:
        print(
            f"\n⚠️  WARNING: Missing letters in dataset: {sorted(missing_letters)}")
        print(
            f"   Model will only train on available letters: {actual_letters}")
    else:
        print(f"\n✅ All A-Z letters present in dataset!")

    # SPLIT DATA DULU SEBELUM PREPROCESSING (mencegah data leakage)
    print("\n📊 Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Preprocess: scale features HANYA dari training data
    n_train, seq_len, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)

    # Fit scaler HANYA pada training data
    scaler = StandardScaler()
    scaler.fit(X_train_flat)

    # Transform training data
    X_train_flat_scaled = scaler.transform(X_train_flat)
    X_train_scaled = X_train_flat_scaled.reshape(n_train, seq_len, n_features)

    # Transform test data menggunakan scaler yang sama
    n_test = len(X_test)
    X_test_flat = X_test.reshape(-1, n_features)
    X_test_flat_scaled = scaler.transform(X_test_flat)
    X_test_scaled = X_test_flat_scaled.reshape(n_test, seq_len, n_features)

    print("✅ Scaling completed (fit on train, transform on train & test)")

    # save scaler
    with open(scaler_name, 'wb') as f:
        pickle.dump(scaler, f)

    # encode labels
    lb = LabelBinarizer()
    y_train_enc = lb.fit_transform(y_train)
    y_test_enc = lb.transform(y_test)

    # ensure one-hot
    if y_train_enc.ndim == 1:
        y_train_enc = tf.keras.utils.to_categorical(
            y_train_enc, num_classes=len(lb.classes_))
        y_test_enc = tf.keras.utils.to_categorical(
            y_test_enc, num_classes=len(lb.classes_))
    elif y_train_enc.shape[1] == 1:
        y_train_enc = tf.keras.utils.to_categorical(
            y_train_enc.ravel(), num_classes=len(lb.classes_))
        y_test_enc = tf.keras.utils.to_categorical(
            y_test_enc.ravel(), num_classes=len(lb.classes_))

    # save label binarizer
    with open(binarizer_name, 'wb') as f:
        pickle.dump(lb, f)

    # Assign to variables expected by training loop
    X_train_raw = X_train_scaled
    X_test_raw = X_test_scaled
    y_train_raw = y_train_enc
    y_test_raw = y_test_enc

    # build LSTM model with regularization
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(
            128, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(len(lb.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

else:
    # --> Load dataset (static CSV)
    df = pd.read_csv(dataset_name)

    # --> Pisahkan fitur dan label
    X = df.drop("label", axis=1).values.astype("float32")
    y = df["label"].values

    # --> SPLIT DATA DULU SEBELUM PREPROCESSING (mencegah data leakage)
    print("\n📊 Splitting data into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # --> Normalisasi HANYA dari training data
    scaler = StandardScaler()
    scaler.fit(X_train)  # Fit hanya pada training
    X_train_scaled = scaler.transform(X_train)  # Transform training
    # Transform test dengan scaler yang sama
    X_test_scaled = scaler.transform(X_test)
    print("✅ Scaling completed (fit on train, transform on train & test)")

    # --> Simpan scaler
    with open(scaler_name, "wb") as f:
        pickle.dump(scaler, f)

    # --> One-hot encoding label
    lb = LabelBinarizer()
    y_train_encoded = lb.fit_transform(y_train)
    y_test_encoded = lb.transform(y_test)

    # Ensure one-hot encoding
    if y_train_encoded.ndim == 2 and y_train_encoded.shape[1] == 1:
        y_train_encoded = tf.keras.utils.to_categorical(
            y_train_encoded.ravel(), num_classes=len(lb.classes_))
        y_test_encoded = tf.keras.utils.to_categorical(
            y_test_encoded.ravel(), num_classes=len(lb.classes_))

    print(
        f"✅ Encoded labels: train shape {y_train_encoded.shape}, test shape {y_test_encoded.shape}")

    # --> Simpan label encoder
    with open(binarizer_name, "wb") as f:
        pickle.dump(lb, f)

    # --> Assign to variables expected by training loop
    X_train_raw = X_train_scaled
    X_test_raw = X_test_scaled
    y_train_raw = y_train_encoded
    y_test_raw = y_test_encoded

    # build MLP model for static data with regularization
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu',
                              input_shape=(X_train_scaled.shape[1],),
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(lb.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

# --> Prepare targets (already encoded properly in both branches)
y_train = y_train_raw
y_test = y_test_raw

print(f"\nFinal data shapes:")
print(f"   X_train: {X_train_raw.shape}, y_train: {y_train.shape}")
print(f"   X_test: {X_test_raw.shape}, y_test: {y_test.shape}")

# --> Callbacks with better overfitting prevention
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        patience=3,
        factor=0.5,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        model_name,
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    )
]

# --> Print model summary
print("\nModel Architecture:")
model.summary()

# --> Train
print("\nStarting training...")
history = model.fit(
    X_train_raw, y_train,
    validation_data=(X_test_raw, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# --> Evaluate
print("\nEvaluating model...")
model.load_weights(model_name)

# Evaluate on training data to check overfitting
train_loss, train_acc = model.evaluate(X_train_raw, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test_raw, y_test, verbose=0)

print(f"\nTraining Accuracy: {round(train_acc * 100, 2)}%")
print(f"Test Accuracy: {round(test_acc * 100, 2)}%")
print(
    f"Gap (overfitting indicator): {round((train_acc - test_acc) * 100, 2)}%")

if (train_acc - test_acc) > 0.05:
    print("WARNING: Model might be overfitting (gap > 5%)")
else:
    print("Model generalization looks good!")

# Predictions
y_pred = model.predict(X_test_raw, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# --> Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=lb.classes_))
acc = np.mean(y_pred_classes == y_true)
print(f"\nAkurasi Test Akhir: {round(acc * 100, 2)}%")

# --> Confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_true, y_pred_classes), annot=True, fmt='d', cmap='Blues',
            xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.title(f'Confusion Matrix (Accuracy: {round(acc * 100, 2)}%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
