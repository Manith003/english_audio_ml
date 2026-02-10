# import librosa
# import numpy as np
# import soundfile as sf
# import os
# import random

# def augment(audio, sr):
#     if random.random() < 0.5:
#         audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.9,1.1))
#     if random.random() < 0.5:
#         audio = audio * random.uniform(0.8,1.2)
#     return audio

# folders = ["audio_data/help","audio_data/noise"]

# for folder in folders:
#     for file in os.listdir(folder):
#         path = os.path.join(folder,file_toggle := file)
#         audio,sr = librosa.load(path,sr=16000)

#         for i in range(3):
#             aug = augment(audio,sr)
#             sf.write(f"{folder}/aug_{i}_{file}",aug,sr)


import librosa
import numpy as np
import soundfile as sf
import os
import random

SR = 16000
WINDOW = 24000

def augment(audio):
    if random.random()<0.5:
        audio = librosa.effects.time_stretch(audio, rate=random.uniform(0.9,1.1))
    if random.random()<0.5:
        audio *= random.uniform(0.8,1.2)
    if random.random()<0.5:
        audio += np.random.randn(len(audio))*0.005
    return audio

folders=["audio_data/help"]

for folder in folders:
    for f in os.listdir(folder):
        audio,_ = librosa.load(os.path.join(folder,f),sr=SR)

        if len(audio)<WINDOW:
            audio=np.pad(audio,(0,WINDOW-len(audio)))
        else:
            audio=audio[:WINDOW]

        for i in range(1):
            aug = augment(audio)
            sf.write(f"{folder}/aug_{i}_{f}",aug,SR)
