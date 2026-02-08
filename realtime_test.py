import sounddevice as sd
import librosa
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("audio_model.keras")

CONFIDENCE = 0.80   # lower for now so we see results

print("🎤 Listening... Say HELP or HELP ME")

while True:
    audio = sd.rec(32000, 16000, 1)
    sd.wait()

    audio = audio.flatten()

    # Ignore silence
    if np.max(np.abs(audio)) < 0.01:
        continue

    mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=64)
    mel = librosa.power_to_db(mel)

    mel = np.resize(mel, (64,64))
    mel = mel[..., np.newaxis]
    mel = np.expand_dims(mel,0)

    prediction = model.predict(mel, verbose=0)[0]

    help_prob = prediction[0]
    noise_prob = prediction[1]

    print("HELP:", round(help_prob,2), "NOISE:", round(noise_prob,2))

    if help_prob > CONFIDENCE:
        print("🧠 HELP DETECTED!")
