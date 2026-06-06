import os

# -------------------------------------------------------
# FORCE GPU USAGE
# -------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU,
    Dense,
    Dropout,
    Input,
    Bidirectional,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import Adam

# -------------------------------------------------------
# GPU CHECK
# -------------------------------------------------------
print("=" * 60)
print("CHECKING GPU")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"\nGPU DETECTED: {gpus}")

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print("\nTensorFlow GPU memory growth enabled")

    except RuntimeError as e:
        print(e)

else:
    print("\nNO GPU DETECTED")
    print("Using CPU instead")

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    'data',
    'large_dataset.csv'
)

MODEL_DIR = os.path.join(BASE_DIR, 'models')

PLOT_DIR = os.path.join(
    BASE_DIR,
    'outputs',
    'plots'
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("\n" + "=" * 60)
print("ADVANCED GPU GRU/LSTM SOLAR STORM PREDICTION")
print("=" * 60)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

# -------------------------------------------------------
# FEATURES
# -------------------------------------------------------
X = df.drop(columns=['storm_label'])
y = df['storm_label']

print(f"Dataset shape: {df.shape}")
print(f"Feature count: {X.shape[1]}")

# -------------------------------------------------------
# SCALE FEATURES
# -------------------------------------------------------
print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------------------------------
# CREATE SEQUENCES
# -------------------------------------------------------
SEQUENCE_LENGTH = 30

X_seq = []
y_seq = []

for i in range(SEQUENCE_LENGTH, len(X_scaled)):
    X_seq.append(X_scaled[i-SEQUENCE_LENGTH:i])
    y_seq.append(y.iloc[i])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.float32)

print(f"\nSequence shape: {X_seq.shape}")

# -------------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.2,
    random_state=42,
    stratify=y_seq
)

print(f"Train sequences: {len(X_train)}")
print(f"Test sequences : {len(X_test)}")

# -------------------------------------------------------
# CLASS WEIGHTS
# -------------------------------------------------------
print("\nComputing class weights...")

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    0: weights[0],
    1: weights[1]
}

print(f"Class Weights: {class_weights}")

# -------------------------------------------------------
# BUILD MODEL
# -------------------------------------------------------
print("\nBuilding Advanced GRU Model...")

with tf.device('/GPU:0'):

    model = Sequential([

        Input(shape=(SEQUENCE_LENGTH, X_train.shape[2])),

        Bidirectional(
            GRU(
                128,
                return_sequences=True
            )
        ),

        BatchNormalization(),

        Dropout(0.3),

        Bidirectional(
            GRU(
                64,
                return_sequences=False
            )
        ),

        BatchNormalization(),

        Dropout(0.3),

        Dense(64, activation='relu'),

        Dropout(0.2),

        Dense(32, activation='relu'),

        Dense(1, activation='sigmoid')
    ])

# -------------------------------------------------------
# COMPILE MODEL
# -------------------------------------------------------
model.compile(
    optimizer=Adam(
        learning_rate=0.0003
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------------------------------
# CALLBACKS
# -------------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    verbose=1
)

# -------------------------------------------------------
# TRAINING
# -------------------------------------------------------
print("\nTraining Advanced LSTM on GPU...")

history = model.fit(
    X_train,
    y_train,
    epochs=60,
    batch_size=256,
    validation_split=0.2,
    callbacks=[
        early_stop,
        reduce_lr
    ],
    class_weight=class_weights,
    verbose=1
)

# -------------------------------------------------------
# PREDICTIONS
# -------------------------------------------------------
print("\nGenerating predictions...")

y_prob = model.predict(X_test).flatten()

y_pred = (y_prob >= 0.5).astype(int)

# -------------------------------------------------------
# METRICS
# -------------------------------------------------------
accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 60)
print("ADVANCED GPU GRU PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"ROC AUC  : {auc:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)

# -------------------------------------------------------
# TRAINING CURVE
# -------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title(
    'Advanced GPU GRU Training Curve'
)

plt.legend()

curve_path = os.path.join(
    PLOT_DIR,
    'advanced_gpu_gru_training_curve.png'
)

plt.savefig(
    curve_path,
    dpi=300,
    bbox_inches='tight'
)

print("\nSaved training curve:")
print(curve_path)

# -------------------------------------------------------
# SAVE MODEL
# -------------------------------------------------------
model_path = os.path.join(
    MODEL_DIR,
    'advanced_gpu_gru_model.keras'
)

model.save(model_path)

print("\nSaved model:")
print(model_path)

print("\n" + "=" * 60)
print("ADVANCED GPU TRAINING COMPLETE")
print("=" * 60)