# import librosa
# import numpy as np
# import os

# X = []
# y = []

# labels = {
#     "help": 0,
#     "noise": 1
# }

# print("Extracting features...")

# for word in labels:
#     folder = f"audio_data/{word}"

#     for file in os.listdir(folder):
#         path = os.path.join(folder, file)

#         audio, sr = librosa.load(path, sr=16000)

#         mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
#         mfcc = np.mean(mfcc.T, axis=0)

#         X.append(mfcc)
#         y.append(labels[word])

# X = np.array(X)
# y = np.array(y)

# np.save("X.npy", X)
# np.save("y.npy", y)

# print("Saved X.npy and y.npy")
# print("Samples:", len(X))


# 

import librosa
import numpy as np
import os

DATA = "audio_data"
LABELS = {"help":0, "noise":1}

SR = 16000
WINDOW = 24000      # 1.5 sec
N_MELS = 64
TARGET_FRAMES = 96

X=[]
y=[]

def load_audio(path):
    audio, _ = librosa.load(path, sr=SR)

    # Force 1.5 sec
    if len(audio) < WINDOW:
        audio = np.pad(audio, (0, WINDOW-len(audio)))
    else:
        audio = audio[:WINDOW]

    return audio

def make_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_mels=N_MELS
    )
    mel = librosa.power_to_db(mel)

    # Safe time resize
    if mel.shape[1] < TARGET_FRAMES:
        mel = np.pad(mel, ((0,0),(0,TARGET_FRAMES-mel.shape[1])))
    else:
        mel = mel[:,:TARGET_FRAMES]

    # Normalize
    mel = (mel - mel.mean()) / (mel.std()+1e-6)

    return mel

for word in LABELS:
    folder = os.path.join(DATA, word)

    for f in os.listdir(folder):
        audio = load_audio(os.path.join(folder,f))
        mel = make_mel(audio)

        X.append(mel)
        y.append(LABELS[word])

X = np.array(X)[...,None]
y = np.array(y)

np.save("X.npy",X)
np.save("y.npy",y)

print("Dataset saved")
print("Samples:",len(X))
