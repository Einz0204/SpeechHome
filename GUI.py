import sounddevice as sd  # 用於錄音
import scipy.io.wavfile as wavfile  # 用於讀寫 WAV 檔案
import librosa  # 用於音訊處理（去靜音、擷取特徵等）
import numpy as np  # 用於數值運算\ import tensorflow as tf  # 用於載入及推論 TensorFlow 模型
import tkinter as tk  # 用於建立 GUI
import tensorflow as tf  # 用於載入及推論 TensorFlow 模型
import time  # 用於計算時間
import threading  # 用於執行非同步的辨識流程
import os  # 用於檔案與路徑操作
from collections import deque  # 雙向佇列，用於滑動視窗緩衝
from scipy.io.wavfile import write  # 快速寫出 WAV 檔案

# === 參數設定 ===
fs_activation    = 16000    # 錄製啟動詞時的取樣率 (Hz)
fs_command       = 16000    # 錄製指令時的取樣率 (Hz)
window_length    = 2        # 滑動視窗長度 (秒)，用於指令辨識時的緩衝
step_duration    = 0.7      # 每次錄音的時間片長度 (秒)
timeout_command  = 20.0     # 最多等待指令的總時長 (秒)
activation_dur   = 3.0      # 錄製啟動詞的時間長度 (秒)
record_file      = 'raw.wav'  # 臨時錄音檔案名稱
model_path       = os.path.join('classifier', 'cnn_model.h5')  # 已訓練模型檔案路徑
test_dir         = 'test_inputs'  # 用於儲存測試及中間檔案的資料夾

# === 信心度 & 標籤設定 ===
CONF_THRESHOLD      = 0.85  # 執行指令的最低信心門檻
EXECUTABLE_COMMANDS = ["開燈", "關燈"]  # 若辨識到其中指令且信心足夠，優先執行
NON_EXECUTABLE_TAGS = ["UNKNOWN", "Noise"]  # 這兩類不管信心多高都不執行

# 確保用來存檔案的資料夾存在
os.makedirs(test_dir, exist_ok=True)

# === 載入所有指令標籤 ===
with open("classifier/labels.txt", "r", encoding="utf-8") as f:
    # labels.txt 每行一個標籤
    COMMANDS = [l.strip() for l in f]

# === 載入模型 (compile=False 可加快載入) ===
MODEL = tf.keras.models.load_model(model_path, compile=False)

# === 功能函數：錄音並寫入檔案 ===
def record_audio_to_file(sr, duration, filename):
    """
    錄製一段音訊，並以 int16 格式存為 WAV。
    sr: 取樣率，duration: 錄音秒數，filename: 輸出檔案
    """
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    wavfile.write(filename, sr, audio)

# === 功能函數：錄製音訊片段並回傳正規化後的 numpy 陣列 ===
def record_chunk(sr, duration):
    """
    錄製一段音訊並回傳 float32 陣列，範圍 [-1, 1]
    """
    audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    return audio.flatten().astype(np.float32) / 32767.0

# === 功能函數：從檔案去除靜音並固定長度 ===
def trim_speech_from_file(fn, sr, target_dur=1.5):
    """
    讀入檔案 fn，去除靜默片段，並 pad 或截斷至 target_dur。
    回傳目標長度的 1D 陣列，若無語音則回傳 None。
    """
    y, _ = librosa.load(fn, sr=sr)
    # 找出非靜音區間
    intervals = librosa.effects.split(y, top_db=20)
    if not len(intervals):
        return None  # 沒語音
    # 合併所有非靜音片段
    s = np.concatenate([y[start:end] for start, end in intervals])
    L = int(target_dur * sr)
    # 截斷或補零
    return s[:L] if len(s) >= L else np.pad(s, (0, L-len(s)))

# === 功能函數：從陣列去除靜音並固定長度 ===
def trim_speech_from_array(y, sr, target_dur=1.5):
    """
    與 trim_speech_from_file 相同，但輸入為已讀入的 y 陣列。
    """
    intervals = librosa.effects.split(y, top_db=20)
    if not len(intervals):
        return None
    s = np.concatenate([y[start:end] for start, end in intervals])
    L = int(target_dur * sr)
    return s[:L] if len(s) >= L else np.pad(s, (0, L-len(s)))

# === 功能函數：補齊或截斷 MFCC 到固定欄數 ===
def fix_mfcc_length(mfcc, target_frames=32):
    """
    將第 2 維 (時間軸) pad 或截斷至 target_frames。
    """
    if mfcc.shape[1] < target_frames:
        pad_amount = target_frames - mfcc.shape[1]
        return np.pad(mfcc, ((0,0),(0,pad_amount)), mode='constant')
    return mfcc[:, :target_frames]

# === 功能函數：模型推論 ===
def predict_command(signal, sr):
    """
    對輸入的音訊訊號 signal 執行 MFCC -> 模型預測 -> 回傳 (label, confidence)
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    mfcc = fix_mfcc_length(mfcc, target_frames=32)
    mfcc = mfcc[..., np.newaxis]  # (n_mfcc, time, 1)
    mfcc = np.expand_dims(mfcc, 0)  # (1, n_mfcc, time, 1)
    pred = MODEL.predict(mfcc, verbose=0)[0]
    idx, conf = int(np.argmax(pred)), float(np.max(pred))
    return COMMANDS[idx], conf

# === 等待啟動詞 ===
def listen_for_activation():
    """
    重複錄音與推論，直到偵測到「啟動」且信心 >= 0.8
    """
    result_var.set("🎧 等待『啟動』…")
    while True:
        record_audio_to_file(fs_activation, activation_dur, record_file)
        sig = trim_speech_from_file(record_file, fs_activation)
        if sig is None:
            continue
        cmd, conf = predict_command(sig, fs_activation)
        if cmd == "啟動" and conf >= 0.8:
            result_var.set("✅ 啟動成功！")
            return

# === 辨識指令（20 秒內滑動視窗） ===
def listen_for_command():
    """
    在 timeout_command 內，每 step_duration 錄一次 chunk，
    以 deque(window_length) 作滑動窗口，持續推論並儲存所有中間檔案。
    回傳最終要執行的 command (或 None)。
    """
    buf = deque(maxlen=int(step_duration * fs_command))  # 滑動窗口緩衝
    full_audio = []  # 收集所有 chunk
    start = time.time()
    chunk_idx = 0  # 原始 chunk 計數
    window_idx = 0  # 滑動窗口計數
    candidate, cand_conf = None, 0.0  # 候選指令與信心

    result_var.set(f"⌛ 辨識開始 (0/{int(timeout_command)}s)")
    while time.time() - start < timeout_command:
        elapsed = time.time() - start
        result_var.set(f"⌛ 辨識中，剩 {timeout_command - elapsed:.1f}s")

        # 錄製 step_duration 長度的原始 chunk
        chunk = record_chunk(fs_command, window_length)
        chunk_idx += 1
        full_audio.append(chunk)

        # 儲存原始 chunk
        chunk_path = os.path.join(test_dir, f"chunk_{chunk_idx:03d}.wav")
        write(chunk_path, fs_command, (chunk * 32767).astype(np.int16))
        print(f"💾 Saved chunk: {chunk_path}")

        # 更新滑動緩衝
        buf.extend(chunk)
        # 緩衝尚未達到 window_length
        if len(buf) < buf.maxlen:
            continue

        # 當緩衝滿，取出一段音訊做 trim、推論
        window_idx += 1
        audio_window = np.array(buf)
        sig = trim_speech_from_array(audio_window, fs_command)
        if sig is None:
            continue

        input_path = os.path.join(test_dir, f"input_{window_idx:03d}.wav")
        write(input_path, fs_command, (sig * 32767).astype(np.int16))
        print(f"💾 Saved input window: {input_path}")

        cmd, conf = predict_command(sig, fs_command)
        print(f"[{elapsed:.1f}s] 偵測 {cmd} (conf={conf:.2f})")

        # 過濾不可執行或信心不足
        if cmd in NON_EXECUTABLE_TAGS or conf < CONF_THRESHOLD:
            continue
        # 優先記錄可執行命令
        if cmd in EXECUTABLE_COMMANDS and candidate is None:
            candidate, cand_conf = cmd, conf
            result_var.set(f"✅ 優先指令：{cmd} ({conf:.2f})")
        elif candidate is None:
            candidate, cand_conf = cmd, conf
            result_var.set(f"✅ 偵測指令：{cmd} ({conf:.2f})")

    # 20 秒結束，儲存完整音訊並回傳結果
    full_array = np.concatenate(full_audio)
    full_path = os.path.join(test_dir, 'full_20s.wav')
    write(full_path, fs_command, (full_array * 32767).astype(np.int16))
    print(f"💾 Saved full segment: {full_path}")

    if candidate:
        result_var.set(f"🚀 執行：{candidate} ({cand_conf:.2f})")
        return candidate

    result_var.set("❌ 時間到，未偵測到有效指令")
    return None

# === 主流程 ===
def process():
    # 先等待啟動詞
    listen_for_activation()
    # 辨識並執行指令
    cmd = listen_for_command()
    if cmd:
        print(f"🚀 執行命令：{cmd}")
    else:
        print("⚠️ 無命令執行")

# 啟動新執行緒以免阻塞 GUI
def start_process():
    threading.Thread(target=process, daemon=True).start()

# === 建立 GUI ===
root = tk.Tk()
root.title("🎤 聲控系統")

result_var = tk.StringVar()  # 用於顯示狀態文字
label = tk.Label(root, textvariable=result_var, font=("Arial", 16), fg="blue")
label.pack(pady=20)

btn = tk.Button(root, text="🎙️ 啟動辨識", font=("Arial", 14), command=start_process)
btn.pack(pady=10)

# 未實作的新增指令按鈕
add_btn = tk.Button(root, text="➕ 新增指令（未實作）", font=("Arial", 12))
add_btn.pack(pady=5)

root.mainloop()  # 進入 GUI 事件迴圈
