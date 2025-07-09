import os
import numpy as np
import random
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

# === 超參數設定 ===
SAMPLE_RATE = 16000
DURATION = 1.0
N_MFCC = 40
MAX_LEN = 32
AUG_PER_SAMPLE = 1  # 每筆資料會產生 3 種方式，每種 1 次，總共會有 1+3*1 = 4 倍資料量

# === 資料路徑 ===
DATA_DIR = "sound"
OUTPUT_DIR = "classifier"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 取得標籤 ===
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
label_to_index = {label: i for i, label in enumerate(labels)}

X, y_all = [], []

# === 特徵萃取 ===
def extract_mfcc(y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, max_len=MAX_LEN):
    y = y[:sr] if len(y) > sr else np.pad(y, (0, sr - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc[:, :max_len] if mfcc.shape[1] >= max_len else np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
    return mfcc.T  # shape: (32, 40)

# === 多重資料增強 ===
def multi_augment_audio(y, sr):
    audios = []
    for _ in range(AUG_PER_SAMPLE):
        audios.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2)))
        audios.append(librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1)))
        audios.append(y + np.random.normal(0, 0.005, len(y)))
    return audios

# === 資料前處理 ===
for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        if not file.endswith(".wav"):
            continue
        path = os.path.join(folder, file)
        try:
            y, sr = librosa.load(path, sr=SAMPLE_RATE)
            mfcc = extract_mfcc(y, sr)
            X.append(mfcc)
            y_all.append(label_to_index[label])
            for aug in multi_augment_audio(y, sr):
                mfcc_aug = extract_mfcc(aug, sr)
                X.append(mfcc_aug)
                y_all.append(label_to_index[label])
        except Exception as e:
            print(f"[❌] {path}: {e}")

# === 資料格式轉換 ===
X = np.array(X)[..., np.newaxis]       # (n, 32, 40, 1)
X = np.transpose(X, (0, 2, 1, 3))       # (n, 40, 32, 1)
y = np.array(y_all)

# === 切分訓練測試資料集 ===
assert len(X) == len(y), "❌ X 和 y 筆數不一致"
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# === CNN 模型架構 + L2 正則化 ===
l2 = tf.keras.regularizers.l2(0.001)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=l2, input_shape=X.shape[1:]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=l2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=l2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(len(labels), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# === 訓練模型 ===
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=3, min_lr=1e-5)

start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)
end_time = time.time()

# === 測試正確率 ===
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ 測試正確率：{test_acc:.4f}")
print(f"🕒 訓練時間：{(end_time - start_time):.2f} 秒")

# === 儲存模型與標籤 ===
model.save(os.path.join(OUTPUT_DIR, "cnn_model.h5"))
with open(os.path.join(OUTPUT_DIR, "labels.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(labels))

# === 繪圖修正 ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')  # ✅ 修正 label
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curve.png"))
plt.close()

print(f"📊 模型與標籤已儲存至：{OUTPUT_DIR}/")
