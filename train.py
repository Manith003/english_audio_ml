# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split

# X = np.load("X.npy")
# y = np.load("y.npy")

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2
# )

# model=tf.keras.Sequential([
#  tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(64,64,1)),
#  tf.keras.layers.MaxPool2D(),
#  tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
#  tf.keras.layers.MaxPool2D(),
#  tf.keras.layers.Flatten(),
#  tf.keras.layers.Dense(64,activation='relu'),
#  tf.keras.layers.Dense(2,activation='softmax')
# ])


# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# print("Training model...")

# model.fit(X_train, y_train, epochs=40)

# loss, acc = model.evaluate(X_test, y_test)

# print("Accuracy:", acc)

# model.save("audio_model.keras")

# print("Model saved as audio_model")


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

X = np.load("X.npy")
y = np.load("y.npy")

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

model = tf.keras.Sequential([
 tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,96,1)),
 tf.keras.layers.MaxPool2D(),
 tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
 tf.keras.layers.MaxPool2D(),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(128,activation='relu'),
 tf.keras.layers.Dropout(0.3),
 tf.keras.layers.Dense(2,activation='softmax')
])

model.compile(
 optimizer='adam',
 loss='sparse_categorical_crossentropy',
 metrics=['accuracy']
)

# Optional class weighting (enable later if needed)
# weights = {0:2.0, 1:1.0}

model.fit(
 X_train,y_train,
 epochs=35,
 batch_size=16
 # class_weight=weights
)

loss,acc = model.evaluate(X_test,y_test)
print("Accuracy:",acc)

model.save("audio_model.keras")
print("Model saved")
