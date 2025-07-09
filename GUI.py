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

# === 參數設定 ===
fs = 16000
duration = 3  # 每次錄音秒數
record_file = 'raw_record.wav'
trimmed_file = 'trimmed_record.wav'
model_path = os.path.join('classifier', 'cnn_model.h5')

# === 載入標籤 ===
with open("classifier/labels.txt", "r", encoding="utf-8") as f:
    COMMANDS = [line.strip() for line in f.readlines()]

# === 錄音 ===
def record_audio():
    print("🎙️ 開始錄音...")
    audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(record_file, fs, audio)
    print("✅ 錄音完成")

# === 去除靜音 + 修剪 ===
def trim_speech(filename, target_duration=1.5):
    y, sr = librosa.load(filename, sr=fs)
    intervals = librosa.effects.split(y, top_db=20)
    if len(intervals) == 0:
        print("⚠️ 沒偵測到語音")
        return None
    speech = np.concatenate([y[start:end] for start, end in intervals])
    target_len = int(target_duration * sr)
    if len(speech) < target_len:
        speech = np.pad(speech, (0, target_len - len(speech)))
    else:
        speech = speech[:target_len]
    return speech

# === MFCC 處理 ===
def fix_mfcc_length(mfcc, target_frames=32):
    current_frames = mfcc.shape[1]
    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc

# === 預測指令 ===
def predict_command(signal):
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=40)
    mfcc = fix_mfcc_length(mfcc, target_frames=32)
    mfcc = mfcc[..., np.newaxis]         # (40, 32, 1)
    mfcc = np.expand_dims(mfcc, axis=0)  # (1, 40, 32, 1)

    model = tf.keras.models.load_model(model_path)
    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction)
    confidence = prediction[0][predicted_index]
    return COMMANDS[predicted_index], confidence

# === 等待「啟動」 ===
def listen_for_activation():
    result_var.set("🎧 等待『啟動』指令...")
    while True:
        record_audio()
        signal = trim_speech(record_file)
        if signal is None:
            continue
        result, conf = predict_command(signal)
        print(f"📡 偵測到：{result}（{conf:.2f}）")
        if result == "啟動" and conf > 0.8:
            result_var.set("✅ 啟動成功，請下達指令...")
            return

# === 辨識 10 秒內的指令 ===
def listen_for_command(timeout=10):
    start_time = time.time()
    while time.time() - start_time < timeout:
        record_audio()
        signal = trim_speech(record_file)
        if signal is None:
            continue
        write(trimmed_file, fs, (signal * 32767).astype(np.int16))
        result, conf = predict_command(signal)
        print(f"🎯 指令：{result}（{conf:.2f}）")
        if result != "啟動" and conf > 0.7:
            result_var.set(f"✅ 指令：{result}\n信心值：{conf:.2f}")
            return result
    result_var.set("⌛ 指令超時，未收到有效輸入")
    return None

# === 主流程 ===
def start_process():
    threading.Thread(target=process).start()

def process():
    listen_for_activation()
    result = listen_for_command(timeout=10)
    if result:
        print(f"🎉 執行指令：{result}")
        # 🔧 這裡可以加上對應指令的動作，如播放音效、操作硬體等
    else:
        print("⚠️ 無有效指令")

# === GUI ===
root = tk.Tk()
root.title("🎤 語音辨識系統")

result_var = tk.StringVar()
label = tk.Label(root, textvariable=result_var, font=("Arial", 16), fg="blue")
label.pack(pady=20)

btn = tk.Button(root, text="🎙️ 啟動語音辨識", font=("Arial", 14), command=start_process)
btn.pack(pady=10)

add_btn = tk.Button(root, text="➕ 新增指令（未實作）", font=("Arial", 12))
add_btn.pack(pady=5)

root.mainloop()
