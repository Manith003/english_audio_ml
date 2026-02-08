import librosa
import numpy as np
import soundfile as sf
import os
import random

def augment(audio, sr):
    if random.random() < 0.5:
        audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.9,1.1))
    if random.random() < 0.5:
        audio = audio * random.uniform(0.8,1.2)
    return audio

folders = ["audio_data/help","audio_data/noise"]

for folder in folders:
    for file in os.listdir(folder):
        path = os.path.join(folder,file_toggle := file)
        audio,sr = librosa.load(path,sr=16000)

        for i in range(3):
            aug = augment(audio,sr)
            sf.write(f"{folder}/aug_{i}_{file}",aug,sr)
