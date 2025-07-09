import os
import time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import librosa

# 參數設定
fs = 16000  # 取樣率
chunk_duration = 2  # 每次錄音的區段長度（秒）
clip_duration = 1.0  # 要裁切的目標語音長度
output_folder = 'clips'
os.makedirs(output_folder, exist_ok=True)

def trim_speech(audio, sr, target_duration=1.0, top_db=20):
    intervals = librosa.effects.split(audio, top_db=top_db)
    clips = []
    for start, end in intervals:
        segment = audio[start:end]
        if len(segment) < int(sr * 0.2):  # 忽略太短的片段（避免雜音）
            continue
        target_len = int(target_duration * sr)
        if len(segment) < target_len:
            segment = np.pad(segment, (0, target_len - len(segment)))
        else:
            segment = segment[:target_len]
        clips.append(segment)
    return clips

def record_loop():
    idx = 1
    print("🎙️ 持續錄音中，按 Ctrl+C 停止")
    try:
        while True:
            print(f"⏺️ 錄製 {chunk_duration} 秒...")
            audio = sd.rec(int(fs * chunk_duration), samplerate=fs, channels=1, dtype='int16')
            sd.wait()
            y = audio.flatten().astype(np.float32) / 32768.0  # 正規化為 float32 [-1,1]
            clips = trim_speech(y, fs, target_duration=clip_duration)
            print(f"🎧 偵測到 {len(clips)} 段語音")
            for clip in clips:
                wav_path = os.path.join(output_folder, f"ENoff-{idx:03d}.wav")
                wavfile.write(wav_path, fs, (clip * 32767).astype(np.int16))
                print(f"✅ 儲存 {wav_path}")
                idx += 1
    except KeyboardInterrupt:
        print("\n🛑 錄音結束")

if __name__ == "__main__":
    record_loop()