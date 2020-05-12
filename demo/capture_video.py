# %%
import os
import cv2
import sys
import numpy as np
from collections import deque
import time
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
from tensorflow import keras
# %%
model = tf.keras.models.load_model(
    '../training/trained_models/LSTM_50epochs_trimmed', compile=True)

# %%
labels = os.listdir('../data/kth-actions/frame_trimmed')
# x = tf.random.normal((1, 16, 160, 160, 3))
# pred = labels[np.argmax(model(x).numpy)]
# print(pred)
# model(x)
# %%
frames_per_prediction = 16
buffer = deque(maxlen=frames_per_prediction)

cap = cv2.VideoCapture(
    '../data/kth-actions/video/handwaving/person22_handwaving_d1_uncomp.avi')
ret, frame = cap.read()
height = frame.shape[0]
width = frame.shape[1]
font = cv2.FONT_HERSHEY_SIMPLEX
# print(ret, frame.shape)
# print(dummy_data.shape)
print(labels)
t = time.time()
while True:
    # time.sleep(0.025)
    ret, frame = cap.read()
    down_sampled_frame = cv2.resize(frame, (160, 160))
    gray = down_sampled_frame.copy()
    # gray = cv2.cvtColor(down_sampled_frame, cv2.COLOR_BGR2GRAY)
    # gray = gray.res (160,160)
    # gray = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
    buffer.append(gray)
    if len(buffer) == 16:
        pred = "I dumb"
        x = np.expand_dims(np.array(buffer), axis=0).astype(np.float32)
        # print(x.shape)
        pred_ind = np.argmax(model(x).numpy()[0])
        pred_label = labels[pred_ind]
        # print()
        # print(np.argmax(model(x).numpy()))
    #     # print(x.dtype)
    #     # pred = labels[np.argmax(model(x).numpy)]
    #     # print(model(x))
    #     # cv2.rectangle(gray, (0, 0), (100, 100), (255, 0, 0), 2)
        cv2.putText(down_sampled_frame, "Prediction: " + pred_label, (2, height + 25),
                    font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
    if time.time() - t > 10:
        break
    cv2.imshow('frame', down_sampled_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
