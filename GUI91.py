import sounddevice as sd
import numpy as np
import tensorflow as tf
import tkinter as tk
import time
import threading
import os
from collections import deque
from scipy.io.wavfile import write
import librosa
import soundfile as sf
import queue

# === 全域參數 ===
fs_command = 16000
fs_activation = 16000
window_length = 2.0
step_duration = 0.2
timeout_command = 20.0
activation_dur = 3.0
record_file = 'raw.wav'
model_path = os.path.join('classifier', 'cnn_model.h5')
test_dir = 'test_inputs'

CONF_THRESHOLD = 0.85
EXECUTABLE_COMMANDS = ["開燈", "關燈"]
NON_EXECUTABLE_TAGS = ["UNKNOWN", "Noise"]

os.makedirs(test_dir, exist_ok=True)

with open("classifier/labels.txt", "r", encoding="utf-8") as f:
    COMMANDS = [l.strip() for l in f]

MODEL = tf.keras.models.load_model(model_path, compile=False)

audio_queue = queue.Queue()
running = False
start_time = None  # 🕒 Producer 開始送資料時會設定
result_var = None  # GUI 顯示用

# === 錄音 ===
def record_chunk(sr, duration):
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    return audio.flatten().astype(np.float32) / 32767.0

# === 語音擷取（VAD） ===
def trim_speech(signal, sr, target_dur=1.0):
    intervals = librosa.effects.split(signal, top_db=30)
    if len(intervals) == 0:
        return None
    y = np.concatenate([signal[start:end] for start, end in intervals])
    L = int(target_dur * sr)
    return y[:L] if len(y) >= L else np.pad(y, (0, L - len(y)))

# === MFCC 修正長度 ===
def fix_mfcc_length(mfcc, target_frames=32):
    if mfcc.shape[1] < target_frames:
        return np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])), mode='constant')
    return mfcc[:, :target_frames]

# === 模型預測 ===
def predict_command(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = fix_mfcc_length(mfcc)
    mfcc = mfcc[..., np.newaxis]
    mfcc = np.expand_dims(mfcc, 0)
    pred = MODEL.predict(mfcc, verbose=0)[0]
    idx, conf = int(np.argmax(pred)), float(np.max(pred))
    return COMMANDS[idx], conf

# === 啟動詞辨識 ===
def listen_for_activation():
    result_var.set("🎧 等待『啟動』…（不計入辨識時間）")
    root.update()
    while True:
        audio = record_chunk(fs_activation, activation_dur)
        write("activation.wav", fs_activation, (audio * 32767).astype(np.int16))
        sig = trim_speech(audio, fs_activation, target_dur=1.5)
        if sig is None:
            continue
        cmd, conf = predict_command(sig, fs_activation)
        if cmd == "啟動" and conf >= 0.3:
            result_var.set("✅ 啟動成功！")
            root.update()
            return

# === Producer（擷取2秒 clip + VAD成1秒）===
def producer_loop():
    global running, start_time
    fs = fs_command
    window_size = int(fs * window_length)
    buffer = deque(maxlen=window_size)
    clip_idx = 0
    input_idx = 0

    # 初始化 buffer
    while len(buffer) < window_size:
        chunk = record_chunk(fs, 0.05)
        buffer.extend(chunk)

    while running:
        if len(buffer) < window_size:
            time.sleep(step_duration)
            continue

        window = np.array(buffer)[-window_size:]
        clip_idx += 1
        clip_path = os.path.join(test_dir, f"clip_{clip_idx:03d}.wav")
        write(clip_path, fs, (window * 32767).astype(np.int16))

        input_sig = trim_speech(window, fs, target_dur=1.0)
        if input_sig is not None:
            input_idx += 1
            input_path = os.path.join(test_dir, f"input_{input_idx:03d}.wav")
            write(input_path, fs, (input_sig * 32767).astype(np.int16))

            if start_time is None:
                start_time = time.time()  # ✅ 真正辨識時間起點

            audio_queue.put((input_sig, input_idx))

        chunk = record_chunk(fs, 0.05)
        buffer.extend(chunk)
        time.sleep(step_duration)

# === Consumer（推論 + 倒數顯示）===
def consumer_loop():
    global running, start_time
    candidate, cand_conf = None, 0.0

    # 等待 producer 設定 start_time
    while start_time is None and running:
        time.sleep(0.01)

    while running or not audio_queue.empty():
        elapsed = time.time() - start_time
        remaining = max(0, timeout_command - elapsed)

        if remaining <= 0:
            break  # ⏱ 時間到，離開 consumer，producer 也會自動結束

        try:
            signal, input_idx = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        result_var.set(f"⌛ 剩餘時間：{remaining:.1f}s")
        root.update()

        cmd, conf = predict_command(signal, fs_command)
        print(f"[{elapsed:.1f}s] 偵測 {cmd} (conf={conf:.2f})")

        if cmd in NON_EXECUTABLE_TAGS or conf < CONF_THRESHOLD:
            continue

        if cmd in EXECUTABLE_COMMANDS and candidate is None:
            candidate, cand_conf = cmd, conf
            result_var.set(f"✅ 優先指令：{cmd} ({conf:.2f})")
            root.update()
        elif candidate is None:
            candidate, cand_conf = cmd, conf
            result_var.set(f"✅ 偵測指令：{cmd} ({conf:.2f})")
            root.update()

    # 收尾
    if candidate:
        result_var.set(f"🚀 執行：{candidate} ({cand_conf:.2f})")
        print(f"🚀 執行命令：{candidate}")
    else:
        result_var.set("❌ 時間到，未偵測到有效指令")
        print("⚠️ 無命令執行")

    running = False

# === 主流程控制 ===
def process():
    global running, start_time, timeout_command

    # 顯示等待啟動詞（主線程）
    result_var.set("🎧 等待『啟動』…（不計入辨識時間）")
    root.update()

    def background_activation():
        listen_for_activation()

        # 啟動成功後，排入主線程處理後續
        def start_after_activation():
            global running, start_time, timeout_command
            result_var.set("⏳ 辨識中…")
            root.update()

            start_time = None
            running = True
            timeout_command = 20.0  # ✅ 在這裡重設倒數秒數

            threading.Thread(target=producer_loop, daemon=True).start()
            threading.Thread(target=consumer_loop, daemon=True).start()

        root.after(100, start_after_activation)

    threading.Thread(target=background_activation, daemon=True).start()

# === GUI 啟動 ===
def start_process():
    threading.Thread(target=process, daemon=True).start()

root = tk.Tk()
root.title("🎤 聲控系統")

result_var = tk.StringVar()
label = tk.Label(root, textvariable=result_var, font=("Arial", 16), fg="blue")
label.pack(pady=20)

btn = tk.Button(root, text="🎙️ 啟動辨識", font=("Arial", 14), command=start_process)
btn.pack(pady=10)

add_btn = tk.Button(root, text="➕ 新增指令（未實作）", font=("Arial", 12))
add_btn.pack(pady=5)

root.mainloop()
