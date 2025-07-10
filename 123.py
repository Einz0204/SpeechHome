import os
import librosa
import soundfile as sf
import numpy as np
import random

# === 超參數 ===
SAMPLE_RATE = 16000
AUG_PER_SAMPLE = 2  # 每種方式做幾次

# === 資料夾 ===
DATA_DIR = "sound1"
AUGMENTED_DIR = "augmented_sound"
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# === 增強函數 ===
def augment_and_save(y, sr, base_filename, save_dir, aug_type, idx):
    if aug_type == 'pitch':
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
    elif aug_type == 'stretch':
        rate = random.uniform(0.9, 1.1)
        y_aug = librosa.effects.time_stretch(y, rate=rate)
        # 裁切或補零到 1 秒（sr 點）
        if len(y_aug) > sr:
            y_aug = y_aug[:sr]
        else:
            y_aug = np.pad(y_aug, (0, sr - len(y_aug)))
    elif aug_type == 'noise':
        y_aug = y + np.random.normal(0, 0.0005, len(y))
    else:
        return

    # 統一儲存為 wav
    aug_filename = f"{os.path.splitext(base_filename)[0]}_{aug_type}{idx}.wav"
    aug_path = os.path.join(save_dir, aug_filename)
    sf.write(aug_path, y_aug, sr)
    print(f"✅ Saved: {aug_path}")

# === 對資料夾內所有檔案進行增強 ===
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    output_label_dir = os.path.join(AUGMENTED_DIR, label)
    os.makedirs(output_label_dir, exist_ok=True)

    for file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file)
        if not os.path.isfile(file_path):
            continue

        # 嘗試讀檔，librosa.load 會自動用 soundfile 或 audioread
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"[❌] 跳過非音訊或不支援檔案：{file_path}，原因：{e}")
            continue

        base_filename = os.path.basename(file_path)

        # 複製原檔（統一轉成 wav）
        original_out = os.path.join(output_label_dir, os.path.splitext(base_filename)[0] + '.wav')
        sf.write(original_out, y, sr)
        print(f"✔ Copied: {original_out}")

        # 進行資料增強
        for aug_type in ['pitch', 'stretch', 'noise']:
            for i in range(1, AUG_PER_SAMPLE + 1):
                augment_and_save(y, sr, base_filename, output_label_dir, aug_type, i)

print("🎉 全部音檔增強完成！")
