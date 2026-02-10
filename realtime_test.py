# import sounddevice as sd
# import librosa
# import numpy as np
# import tensorflow as tf

# model = tf.keras.models.load_model("audio_model.keras")

# CONFIDENCE = 0.80   # lower for now so we see results

# print("🎤 Listening... Say HELP or HELP ME")

# while True:
#     audio = sd.rec(32000, 16000, 1)
#     sd.wait()

#     audio = audio.flatten()

#     # Ignore silence
#     if np.max(np.abs(audio)) < 0.01:
#         continue

#     mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64)
#     mel = librosa.power_to_db(mel)

#     mel = np.resize(mel, (64,64))
#     mel = mel[..., np.newaxis]
#     mel = np.expand_dims(mel,0)

#     prediction = model.predict(mel, verbose=0)[0]

#     help_prob = prediction[0]
#     noise_prob = prediction[1]

#     print("HELP:", round(help_prob,2), "NOISE:", round(noise_prob,2))

#     if help_prob > CONFIDENCE:
#         print("🧠 HELP DETECTED!")


import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf

SR = 16000
WINDOW = 24000
N_MELS = 64
TARGET = 96
CONF = 0.75

model = tf.keras.models.load_model("audio_model.keras")

def preprocess(audio):
    if len(audio)<WINDOW:
        audio=np.pad(audio,(0,WINDOW-len(audio)))
    else:
        audio=audio[:WINDOW]

    mel = librosa.feature.melspectrogram(y=audio,sr=SR,n_mels=N_MELS)
    mel = librosa.power_to_db(mel)

    if mel.shape[1]<TARGET:
        mel=np.pad(mel,((0,0),(0,TARGET-mel.shape[1])))
    else:
        mel=mel[:,:TARGET]

    mel=(mel-mel.mean())/(mel.std()+1e-6)

    mel=mel[...,None]
    mel=np.expand_dims(mel,0)

    return mel

print("🎤 Listening (1.5 sec windows)...")

while True:
    audio = sd.rec(WINDOW,SR,1)
    sd.wait()

    audio = audio.flatten()

    if np.max(np.abs(audio))<0.01:
        continue

    mel = preprocess(audio)
    p = model.predict(mel,verbose=0)[0]

    print("HELP:",round(p[0],2),"NOISE:",round(p[1],2))

    if p[0]>CONF:
        print("🚨 HELP DETECTED")
