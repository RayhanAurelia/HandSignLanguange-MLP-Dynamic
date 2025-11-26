import os
import pickle, numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, seaborn as sns

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
    for _, row in df_seq.iterrows():
        fname = row['filename']
        label = row['label']
        fpath = os.path.join(sequences_dir, fname)
        if not os.path.exists(fpath):
            print(f"Warning: sequence file not found {fpath}, skipping")
            continue
        arr = np.load(fpath)
        X_list.append(arr.astype('float32'))
        y_list.append(label)

    if len(X_list) == 0:
        raise RuntimeError('No sequence data found.')

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
            raise RuntimeError(f"Feature dimension mismatch: sequences have {n_features} features but CSV has {n_features_stat} features")

        X_stat_seq = np.tile(X_stat[:, None, :], (1, seq_len, 1))
        # combine
        X = np.concatenate([X_seq, X_stat_seq], axis=0)
        y = np.concatenate([y_seq, y_stat], axis=0)
    else:
        X = X_seq
        y = y_seq

    # Preprocess: scale features across all time-steps
    n_samples, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(n_samples, seq_len, n_features)

    # save scaler
    with open(scaler_name, 'wb') as f:
        pickle.dump(scaler, f)

    # encode labels
    lb = LabelBinarizer()
    y_enc = lb.fit_transform(y)
    # ensure one-hot
    if y_enc.ndim == 1:
        y_enc = tf.keras.utils.to_categorical(y_enc, num_classes=len(lb.classes_))
    elif y_enc.shape[1] == 1:
        y_enc = tf.keras.utils.to_categorical(y_enc.ravel(), num_classes=len(lb.classes_))

    # save label binarizer
    with open(binarizer_name, 'wb') as f:
        pickle.dump(lb, f)

    # split
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_scaled, y_enc, test_size=0.2, stratify=y, random_state=42
    )

    # build LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, n_features)),
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(lb.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

else:
    #--> Load dataset (static CSV)
    df = pd.read_csv(dataset_name)

    #--> Pisahkan fitur dan label
    X = df.drop("label", axis=1).values.astype("float32")
    y = df["label"].values

    #--> Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
     
    #--> Simpan scaler
    with open(scaler_name, "wb") as f:
        pickle.dump(scaler, f)

    #--> One-hot encoding label
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)

    #--> Simpan label encoder
    with open(binarizer_name, "wb") as f:
        pickle.dump(lb, f)

    #--> Split stratified
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # build MLP model for static data
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(lb.classes_), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#--> Prepare targets depending on mode
if is_dynamic:
    # y_train_raw / y_test_raw already one-hot from split
    y_train = y_train_raw
    y_test = y_test_raw
else:
    # static: y_train_raw/y_test_raw are label strings, transform to one-hot
    y_train = lb.transform(y_train_raw)
    y_test = lb.transform(y_test_raw)

    # Jika LabelBinarizer mengembalikan kolom tunggal (binary), konversi ke one-hot
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = tf.keras.utils.to_categorical(y_train.ravel(), num_classes=len(lb.classes_))
        y_test = tf.keras.utils.to_categorical(y_test.ravel(), num_classes=len(lb.classes_))



#--> Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(model_name, save_best_only=True, monitor='val_accuracy')
]

#--> Train
history = model.fit(
    X_train_raw, y_train,
    validation_data=(X_test_raw, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

#--> Evaluate
model.load_weights(model_name)
y_pred = model.predict(X_test_raw)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

#--> Report
print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=lb.classes_))
acc = np.mean(y_pred_classes == y_true)
print(f"\n✅ Akurasi Akhir: {round(acc * 100, 2)}%")

#--> Confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(confusion_matrix(y_true, y_pred_classes), annot=True, fmt='d', cmap='Blues',
            xticklabels=lb.classes_, yticklabels=lb.classes_)
plt.title(f'Confusion Matrix (Accuracy: {round(acc * 100, 2)}%)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()