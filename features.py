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


import librosa
import numpy as np
import os

X=[]
y=[]

labels={"help":0,"noise":1}

for word in labels:
    for file in os.listdir(f"audio_data/{word}"):
        audio,sr=librosa.load(f"audio_data/{word}/{file}",sr=16000)
        mel=librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=64)
        mel=librosa.power_to_db(mel)
        mel=np.resize(mel,(64,64))
        X.append(mel)
        y.append(labels[word])

X=np.array(X)[...,None]
y=np.array(y)

np.save("X.npy",X)
np.save("y.npy",y)

print("Saved X.npy and y.npy")
print("Samples:", len(X))