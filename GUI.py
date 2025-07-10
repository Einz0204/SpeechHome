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
fs_activation = 16000        # 啟動詞階段取樣率
fs_command    = 32000        # 指令辨識階段取樣率
duration      = 3            # 每次錄音秒數
record_file   = 'raw_record.wav'
trimmed_file  = 'trimmed_record.wav'
model_path    = os.path.join('classifier', 'cnn_model.h5')

# === 信心度 & 標籤設定 ===
CONF_THRESHOLD       = 0.85
EXECUTABLE_COMMANDS  = ["開燈", "關燈"]
NON_EXECUTABLE_TAGS  = ["UNKNOWN", "Noise"]

# === 載入指令標籤 ===
with open("classifier/labels.txt", "r", encoding="utf-8") as f:
    COMMANDS = [line.strip() for line in f.readlines()]

# === 全域一次載入模型（跳過 compile，以消除 warning 並加快推論） ===
MODEL = tf.keras.models.load_model(model_path, compile=False)

# === 通用錄音函式（可指定取樣率） ===
def record_audio(sr):
    print(f"🎙️ 錄音中（{sr//1000} kHz, {duration}s）...")
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(record_file, sr, audio)
    print("✅ 錄音完成")

# === 去除靜音並固定長度 ===
def trim_speech(filename, sr, target_duration=1.5):
    y, _ = librosa.load(filename, sr=sr)
    intervals = librosa.effects.split(y, top_db=20)
    if len(intervals) == 0:
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
    if mfcc.shape[1] < target_frames:
        pad_width = target_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :target_frames]
    return mfcc

# === 模型預測（改用全域 MODEL） ===
def predict_command(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = fix_mfcc_length(mfcc, target_frames=32)
    mfcc = mfcc[..., np.newaxis]         # (40,32,1)
    mfcc = np.expand_dims(mfcc, axis=0)  # (1,40,32,1)

    pred = MODEL.predict(mfcc)
    idx  = np.argmax(pred)
    conf = float(pred[0][idx])
    return COMMANDS[idx], conf

# === 等待「啟動」關鍵詞 ===
def listen_for_activation():
    result_var.set("🎧 等待『啟動』指令...")
    while True:
        record_audio(fs_activation)
        signal = trim_speech(record_file, fs_activation)
        if signal is None:
            continue
        result, conf = predict_command(signal, fs_activation)
        print(f"📡 偵測到 [{result}] 置信度: {conf:.2f}")
        if result == "啟動" and conf >= 0.8:
            result_var.set("✅ 啟動成功，請下指令...")
            return

# === 10 秒內辨識命令 ===
def listen_for_command(timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        record_audio(fs_command)
        signal = trim_speech(record_file, fs_command)
        if signal is None:
            result_var.set("我沒聽清楚，請再說一次")
            continue

        # 儲存裁剪後音檔（非必要，可供除錯）
        write(trimmed_file, fs_command, (signal * 32767).astype(np.int16))

        result, conf = predict_command(signal, fs_command)
        print(f"🎯 偵測 [{result}] 置信度: {conf:.2f}")

        # 非執行標籤
        if result in NON_EXECUTABLE_TAGS:
            result_var.set("我沒聽清楚，請再說一次")
            continue
        # 信心度門檻
        if conf < CONF_THRESHOLD:
            result_var.set("我沒聽清楚，請再說一次")
            continue
        # 優先執行開/關燈
        if result in EXECUTABLE_COMMANDS:
            result_var.set(f"✅ 指令：{result}\n信心值：{conf:.2f}")
            return result
        # 其他有效命令
        result_var.set(f"✅ 指令：{result}\n信心值：{conf:.2f}")
        return result

    result_var.set("⌛ 指令超時，未收到有效輸入")
    return None

# === 主流程 ===
def process():
    listen_for_activation()
    cmd = listen_for_command(timeout=10)
    if cmd:
        print(f"🚀 執行命令：{cmd}")
        # 在此加入對應命令的實際動作
    else:
        print("⚠️ 無有效指令")

def start_process():
    threading.Thread(target=process, daemon=True).start()

# === GUI ===
root = tk.Tk()
root.title("🎤 聲控系統")

result_var = tk.StringVar()
label = tk.Label(root, textvariable=result_var, font=("Arial", 16), fg="blue")
label.pack(pady=20)

btn = tk.Button(root, text="🎙️ 啟動語音辨識", font=("Arial", 14), command=start_process)
btn.pack(pady=10)

add_btn = tk.Button(root, text="➕ 新增指令（未實作）", font=("Arial", 12))
add_btn.pack(pady=5)

root.mainloop()
