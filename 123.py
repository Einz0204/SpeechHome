import os
import librosa
import soundfile as sf
import numpy as np
import random

# === 超參數 ===
SAMPLE_RATE = 16000
AUG_PER_SAMPLE = 2  # 每種方式幾次

# === 資料夾 ===
DATA_DIR = "sound1"
AUGMENTED_DIR = "augmented_sound"
os.makedirs(AUGMENTED_DIR, exist_ok=True)

# === 增強函數 ===
def augment_and_save(y, sr, save_path, aug_type, idx):
    if aug_type == 'pitch':
        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=random.uniform(-2, 2))
    elif aug_type == 'stretch':
        rate = random.uniform(0.9, 1.1)
        y_aug = librosa.effects.time_stretch(y, rate=rate)
        if len(y_aug) > sr:
            y_aug = y_aug[:sr]
        else:
            y_aug = np.pad(y_aug, (0, sr - len(y_aug)))
    elif aug_type == 'noise':
        y_aug = y + np.random.normal(0, 0.0005, len(y))
    else:
        return  # invalid type

    aug_filename = f"{os.path.splitext(os.path.basename(save_path))[0]}_{aug_type}{idx}.wav"
    aug_path = os.path.join(os.path.dirname(save_path), aug_filename)
    sf.write(aug_path, y_aug, sr)
    print(f"✅ Saved: {aug_path}")

# === 對資料夾內所有 wav 複雜化 ===
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    output_label_dir = os.path.join(AUGMENTED_DIR, label)
    os.makedirs(output_label_dir, exist_ok=True)

    for file in os.listdir(label_dir):
        if not file.endswith(".wav"):
            continue

        input_path = os.path.join(label_dir, file)
        try:
            y, sr = librosa.load(input_path, sr=SAMPLE_RATE)
            output_path = os.path.join(output_label_dir, file)

            # 複製原檔
            sf.write(output_path, y, sr)
            print(f"✔ Copied: {output_path}")

            # 進行資料增強
            for aug_type in ['pitch', 'stretch', 'noise']:
                for i in range(AUG_PER_SAMPLE):
                    augment_and_save(y, sr, output_path, aug_type, i+1)

        except Exception as e:
            print(f"[❌] {input_path}: {e}")

print("🎉 全部音檔增強完成！")
