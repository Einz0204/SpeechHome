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

# === 裝置選擇 GUI ===
def select_input_device():
    devices = sd.query_devices()
    input_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]

    if not input_devices:
        raise RuntimeError("未偵測到任何可用的錄音裝置。")

    device_window = tk.Tk()
    device_window.title("🎙️ 選擇錄音裝置")

    selected_index = tk.IntVar(master=device_window)
    selected_index.set(input_devices[0][0])

    def confirm_selection():
        index = selected_index.get()
        print(f"✅ 使用者選擇了裝置 {index}: {devices[index]['name']}")
        sd.default.device = (index, None)
        device_window.destroy()

    label = tk.Label(device_window, text="請選擇要使用的錄音裝置：", font=("Arial", 14))
    label.pack(pady=10)

    for idx, name in input_devices:
        tk.Radiobutton(device_window, text=f"{idx}: {name}", variable=selected_index, value=idx, font=("Arial", 12)).pack(anchor="w")

    btn = tk.Button(device_window, text="✅ 確認", font=("Arial", 14), command=confirm_selection)
    btn.pack(pady=10)

    device_window.mainloop()

# 呼叫裝置選擇
select_input_device()

# === 全域參數設定 ===
fs_command = 16000
fs_activation = 16000
window_length = 2.0
step_duration = 0.2
timeout_command = 20.0
activation_dur = 3.0
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
start_time = None
result_var = None

# === Trim speech ===
def trim_speech(signal, sr, target_dur=1.0):
    intervals = librosa.effects.split(signal, top_db=30)
    if len(intervals) == 0:
        return None
    y = np.concatenate([signal[start:end] for start, end in intervals])
    L = int(target_dur * sr)
    return y[:L] if len(y) >= L else np.pad(y, (0, L - len(y)))

# === 模型預測 ===
def predict_command(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = mfcc[:, :32] if mfcc.shape[1] > 32 else np.pad(mfcc, ((0, 0), (0, 32 - mfcc.shape[1])), mode='constant')
    mfcc = mfcc[..., np.newaxis]
    mfcc = np.expand_dims(mfcc, 0)
    pred = MODEL.predict(mfcc, verbose=0)[0]
    idx, conf = int(np.argmax(pred)), float(np.max(pred))
    return COMMANDS[idx], conf

# === 啟動詞監聽（含 VAD） ===
def listen_for_activation():
    global result_var
    result_var.set("🎧 等待『啟動』…")
    while True:
        print("🎙️ 開始錄音 (啟動詞)...")
        audio = sd.rec(int(fs_activation * activation_dur), samplerate=fs_activation, channels=1, dtype='int16', blocking=True)
        sd.wait()
        audio = audio.flatten().astype(np.float32) / 32767.0
        audio *= 5.0
        audio = np.clip(audio, -1.0, 1.0)
        print(f"[record_chunk] max={np.max(audio):.4f}, min={np.min(audio):.4f}, mean={np.mean(audio):.4f}")

        write("activation.wav", fs_activation, (audio * 32767).astype(np.int16))
        sf.write('output.wav', audio, fs_activation)
        np.save("activation_array.npy", audio)

        sig = trim_speech(audio, fs_activation)
        if sig is None:
            continue

        cmd, conf = predict_command(sig, fs_activation)
        print(f"🔍 啟動詞辨識：{cmd} (conf={conf:.2f})")
        if cmd == "啟動" and conf >= 0.3:
            result_var.set("✅ 啟動成功！")
            sd.stop()
            time.sleep(0.5)
            return

# === InputStream Producer ===
def producer_loop():
    global running
    buffer = deque(maxlen=int(fs_command * window_length * 2))
    clip_idx, input_idx = 0, 0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[Stream status] {status}")
        buffer.extend(indata[:, 0])

    with sd.InputStream(samplerate=fs_command, channels=1, callback=audio_callback):
        print("🔊 錄音串流啟動...")
        time.sleep(2.0)
        start = time.time()

        while running and time.time() - start < timeout_command:
            if len(buffer) < int(fs_command * window_length):
                time.sleep(0.05)
                continue

            clip = np.array(buffer)[-int(fs_command * window_length):]
            clip_path = os.path.join(test_dir, f"clip_{clip_idx:03d}.wav")
            write(clip_path, fs_command, (clip * 32767).astype(np.int16))
            clip_idx += 1

            processed = trim_speech(clip, fs_command, target_dur=1.0)
            if processed is not None:
                input_path = os.path.join(test_dir, f"input_{input_idx:03d}.wav")
                write(input_path, fs_command, (processed * 32767).astype(np.int16))
                audio_queue.put((processed, input_idx))
                input_idx += 1

            time.sleep(step_duration)

    running = False

# === 模型推論 Consumer ===
def consumer_loop():
    global running, start_time
    start_time = time.time()
    candidate, cand_conf = None, 0.0

    while running or not audio_queue.empty():
        try:
            audio_window, input_idx = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        elapsed = time.time() - start_time
        result_var.set(f"⌛ 剩餘時間：{max(0, timeout_command - elapsed):.1f}s")

        cmd, conf = predict_command(audio_window, fs_command)
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

    running = False

# === 啟動辨識流程 ===
def process():
    global running
    listen_for_activation()
    running = True
    result_var.set("⏳ 辨識中…")
    root.update()

    threading.Thread(target=producer_loop, daemon=True).start()
    threading.Thread(target=consumer_loop, daemon=True).start()

# === GUI ===
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
