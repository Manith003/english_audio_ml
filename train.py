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


# import numpy as np
# import tensorflow as tf
# from sklearn.model_selection import train_test_split

# X = np.load("X.npy")
# y = np.load("y.npy")

# X_train,X_test,y_train,y_test = train_test_split(
#     X,y,test_size=0.2,stratify=y,random_state=42
# )

# model = tf.keras.Sequential([
#  tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(64,96,1)),
#  tf.keras.layers.MaxPool2D(),
#  tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#  tf.keras.layers.MaxPool2D(),
#  tf.keras.layers.Flatten(),
#  tf.keras.layers.Dense(128,activation='relu'),
#  tf.keras.layers.Dropout(0.3),
#  tf.keras.layers.Dense(2,activation='softmax')
# ])

# model.compile(
#  optimizer='adam',
#  loss='sparse_categorical_crossentropy',
#  metrics=['accuracy']
# )

# # Optional class weighting (enable later if needed)
# # weights = {0:2.0, 1:1.0}

# model.fit(
#  X_train,y_train,
#  epochs=35,
#  batch_size=16
# #  class_weight=weights
# )

# loss,acc = model.evaluate(X_test,y_test)
# print("Accuracy:",acc)

# model.save("audio_model.keras")
# print("Model saved")


import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
X = np.load("X.npy")
y = np.load("y.npy")

# Stratified split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

# Compute class weights (HELP more important)
classes = np.unique(y_train)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))
print("Class Weights:", class_weights)

# CNN Model
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

# Train with class weighting
model.fit(
 X_train,y_train,
 epochs=35,
 batch_size=16,
 class_weight=class_weights
)

# Evaluate
loss,acc = model.evaluate(X_test,y_test)
print("\nAccuracy:",acc)

# Predictions
pred_probs = model.predict(X_test)
pred = np.argmax(pred_probs,axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test,pred)
print("\nConfusion Matrix (default argmax):")
print(cm)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test,pred,target_names=["HELP","NOISE"]))

# Threshold tuning
print("\nThreshold tuning:")

for t in np.arange(0.5,0.95,0.05):
    custom_pred=[]
    for p in pred_probs:
        if p[0]>t:
            custom_pred.append(0)
        else:
            custom_pred.append(1)

    cm_t = confusion_matrix(y_test,custom_pred)
    print(f"\nThreshold {round(t,2)}")
    print(cm_t)

# Save model
model.save("audio_model.keras")
print("\nModel saved as audio_model.keras")
