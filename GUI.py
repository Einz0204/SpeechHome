import sounddevice as sd
import scipy.io.wavfile as wavfile
import librosa
import numpy as np
import tensorflow as tf
import tkinter as tk
import time
import threading
import os
import queue
from collections import deque
from scipy.io.wavfile import write

# === 參數設定 ===
fs_activation    = 16000
fs_command       = 16000
window_length    = 2.0
step_duration    = 0.2
timeout_command  = 20.0
activation_dur   = 3.0
record_file      = 'raw.wav'
model_path       = os.path.join('classifier', 'cnn_model.h5')
test_dir         = 'test_inputs'

CONF_THRESHOLD      = 0.85
EXECUTABLE_COMMANDS = ["開燈", "關燈"]
NON_EXECUTABLE_TAGS = ["UNKNOWN", "Noise"]

os.makedirs(test_dir, exist_ok=True)

with open("classifier/labels.txt", "r", encoding="utf-8") as f:
    COMMANDS = [l.strip() for l in f]

MODEL = tf.keras.models.load_model(model_path, compile=False)

audio_queue = queue.Queue()
running = False

def trim_speech_from_array(y, sr, target_dur=1.0):
    intervals = librosa.effects.split(y, top_db=30)
    if not len(intervals):
        return None
    s = np.concatenate([y[start:end] for start, end in intervals])
    L = int(target_dur * sr)
    return s[:L] if len(s) >= L else np.pad(s, (0, L - len(s)))

def record_audio_to_file(sr, duration, filename):
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(filename, sr, audio)

def record_chunk(sr, duration):
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    return audio.flatten().astype(np.float32) / 32767.0

def fix_mfcc_length(mfcc, target_frames=32):
    if mfcc.shape[1] < target_frames:
        pad_amount = target_frames - mfcc.shape[1]
        return np.pad(mfcc, ((0,0),(0,pad_amount)), mode='constant')
    return mfcc[:, :target_frames]

def predict_command(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = fix_mfcc_length(mfcc, target_frames=32)
    mfcc = mfcc[..., np.newaxis]
    mfcc = np.expand_dims(mfcc, 0)
    pred = MODEL.predict(mfcc, verbose=0)[0]
    idx, conf = int(np.argmax(pred)), float(np.max(pred))
    return COMMANDS[idx], conf

def listen_for_activation():
    result_var.set("🎧 等待『啟動』…")
    while True:
        record_audio_to_file(fs_activation, activation_dur, record_file)
        y, _ = librosa.load(record_file, sr=fs_activation)

        # 加入 VAD 去靜音並裁成 1 秒
        sig = trim_speech_from_array(y, fs_activation, target_dur=1.0)
        if sig is None:
            continue

        cmd, conf = predict_command(sig, fs_activation)
        if cmd == "啟動" and conf >= 0.8:
            result_var.set("✅ 啟動成功！")
            return

def producer_loop():
    full_audio = []
    buffer = deque(maxlen=int(fs_command * 10))  # 最多保留10秒音訊
    start = time.time()
    next_step = 0.2  # 下一個推論的時間間隔
    last_emit = 0

    while running:
        chunk = record_chunk(fs_command, 0.05)  # 每次錄製50ms（平滑更新）
        buffer.extend(chunk)
        full_audio.append(chunk)

        now = time.time() - start
        if now < 2.0:
            continue  # 前2秒不輸出

        if now - last_emit >= next_step:
            last_emit = now
            if len(buffer) >= int(fs_command * window_length):
                # 從 buffer 回頭取 2 秒
                backtrack = list(buffer)[-int(fs_command * window_length):]
                audio_window = np.array(backtrack)
                audio_queue.put(audio_window)


    
        
    # 儲存整段音訊
    full_path = os.path.join(test_dir, 'full_20s.wav')
    full_array = np.concatenate(full_audio)
    write(full_path, fs_command, (full_array * 32767).astype(np.int16))
    print(f"💾 Saved full segment: {full_path}")

def consumer_loop():
    window_idx = 0
    candidate, cand_conf = None, 0.0
    start = time.time()

    while running and (time.time() - start < timeout_command or not audio_queue.empty()):
        elapsed = time.time() - start

        # ✅ 顯示倒數時間
        if elapsed < timeout_command:
            remaining = timeout_command - elapsed
            result_var.set(f"⌛ 辨識中，剩下 {remaining:.1f}s")

        try:
            sig = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        window_idx += 1
        input_path = os.path.join(test_dir, f"input_{window_idx:03d}.wav")
        write(input_path, fs_command, (sig * 32767).astype(np.int16))
        print(f"💾 Saved input window: {input_path}")

        cmd, conf = predict_command(sig, fs_command)
        print(f"[{elapsed:.1f}s] 偵測 {cmd} (conf={conf:.2f})")

        if cmd in NON_EXECUTABLE_TAGS or conf < CONF_THRESHOLD:
            continue
        if cmd in EXECUTABLE_COMMANDS and candidate is None:
            candidate, cand_conf = cmd, conf
            result_var.set(f"✅ 優先指令：{cmd} ({conf:.2f})")
        elif candidate is None:
            candidate, cand_conf = cmd, conf
            result_var.set(f"✅ 偵測指令：{cmd} ({conf:.2f})")

    if candidate:
        result_var.set(f"🚀 執行：{candidate} ({cand_conf:.2f})")
        print(f"🚀 執行命令：{candidate}")
    else:
        result_var.set("❌ 時間到，未偵測到有效指令")
        print("⚠️ 無命令執行")

def process():
    global running
    listen_for_activation()

    result_var.set("⌛ 辨識中...")
    running = True
    t1 = threading.Thread(target=producer_loop, daemon=True)
    t2 = threading.Thread(target=consumer_loop, daemon=True)
    t1.start()
    t2.start()

    t1.join()
    t2.join()
    running = False

def start_process():
    threading.Thread(target=process, daemon=True).start()

# === GUI ===
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
