import sounddevice as sd
import scipy.io.wavfile as wavfile
import librosa
import numpy as np
import tensorflow as tf
import tkinter as tk
import time
import threading
import os
from scipy.io.wavfile import write

# === 設定參數 ===
fs = 16000
duration = 3  # 錄音秒數
record_file = 'raw_record.wav'
trimmed_file = 'trimmed_record.wav'
model_path = os.path.join('classifier', 'cnn_model.h5')

# === 載入指令標籤 ===
with open("classifier/labels.txt", "r", encoding="utf-8") as f:
    COMMANDS = [line.strip() for line in f.readlines()]

# === 錄音函式 ===
def record_audio():
    print("🎙️ 準備錄音，請在 3 秒內說出指令")
    time.sleep(0)
    audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(record_file, fs, audio)
    print("✅ 錄音完成")

# === 去除靜音區，裁剪語音長度 ===
def trim_speech(filename, target_duration=1.5):
    y, sr = librosa.load(filename, sr=fs)
    intervals = librosa.effects.split(y, top_db=20)  # 靜音分段
    if len(intervals) == 0:
        print("⚠️ 沒有偵測到語音，請再試一次")
        return None
    speech = np.concatenate([y[start:end] for start, end in intervals])
    target_len = int(target_duration * sr)
    if len(speech) < target_len:
        speech = np.pad(speech, (0, target_len - len(speech)))
    else:
        speech = speech[:target_len]
    return speech

# === 補齊 MFCC 長度 ===
def fix_mfcc_length(mfcc, target_frames=32):
    current_frames = mfcc.shape[1]
    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc

# === 模型預測指令 ===
def predict_command(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=40)
    mfcc = fix_mfcc_length(mfcc, target_frames=32)
    mfcc = mfcc[..., np.newaxis]         # shape: (40, 32, 1)
    mfcc = np.expand_dims(mfcc, axis=0)  # shape: (1, 40, 32, 1)

    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]
    return COMMANDS[predicted_index], confidence

# === 執行錄音與預測流程（for GUI）===
def start_process():
    threading.Thread(target=process).start()

def process():
    record_audio()
    signal = trim_speech(record_file)
    if signal is None:
        result_var.set("⚠️ 沒有偵測到語音")
        return

    # 儲存裁剪後音檔
    write(trimmed_file, fs, (signal * 32767).astype(np.int16))
    print(f"💾 已儲存裁剪後音訊：{trimmed_file}")

    result, conf = predict_command(signal)
    result_text = f"🔊 指令：{result}\n信心值：{conf:.2f}"
    print(f"\n🔊 預測結果：{result}（信心值：{conf:.2f}）")
    result_var.set(result_text)

# === GUI 介面 ===
root = tk.Tk()
root.title("🎤 語音辨識系統")

result_var = tk.StringVar()
label = tk.Label(root, textvariable=result_var, font=("Arial", 16), fg="blue")
label.pack(pady=20)

btn = tk.Button(root, text="🎙️ 開始錄音", font=("Arial", 14), command=start_process)
btn.pack(pady=10)

add_btn = tk.Button(root, text="➕ 新增指令（未實作）", font=("Arial", 12))
add_btn.pack(pady=5)

root.mainloop()

