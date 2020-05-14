# %%
import os
import cv2
import sys
import numpy as np
from collections import deque
import time
np.set_printoptions(threshold=sys.maxsize)
import tensorflow as tf
# from tensorflow import keras
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

cap_links = [
    '../data/kth-actions/video/handclapping/person01_handclapping_d1_uncomp.avi',
    '../data/kth-actions/video/running/person01_running_d1_uncomp.avi',
    '../data/kth-actions/video/handwaving/person01_handwaving_d3_uncomp.avi',
    '../data/kth-actions/video/boxing/person23_boxing_d4_uncomp.avi',
    '../data/kth-actions/video/walking/person02_walking_d3_uncomp.avi',
    './videos/2_of_each.avi'
]

cap = cv2.VideoCapture(cap_links[4])

# cap = cv2.VideoCapture(0)
print(labels)

IMG_SIZE = 160  # All images will be resized to 160x160
IMG_SHAPE = [IMG_SIZE, IMG_SIZE, 3]
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("length", length)


def format_example(image):
    image = tf.repeat(image, 3, axis=2) / 255
    image = tf.image.resize(image, IMG_SHAPE[0:2])
    # image = tf.image.per_image_standardization(image)
    # image.set_shape([None] + IMG_SHAPE)
    return image


t = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX
ms_counter = time.time()
buffer_counter = 0
max_frames = 16
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
curr_frame_nr = 0
ms_per_frame = 100
net_input = np.zeros(([1, max_frames] + IMG_SHAPE))
ret = True
pred_label = "Not enought data"
c = 0
writer = None

while True:
    ret, frame = cap.read()
    # if making a prediciont every 100ms then sleep must happen must be turned on
    time.sleep(0.025)
    if ret:
        # down_sampled_frame = cv2.resize(frame, (160, 160))
        # curr_frame_nr = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) + 1
        # print(curr_frame_nr, n_frames)
        # gray = down_sampled_frame.copy()
        # # gray = gray.res (160,160)
        # # gray = np.repeat(gray[:, :, np.newaxis], 3, axis=2)
        if time.time() - ms_counter > 0.1:
            # print(time.time() - ms_counter)
            ms_counter = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = np.expand_dims(gray, 2)
            gray_reshaped = format_example(gray)
            # print(buffer_counter)
            net_input[0, buffer_counter, :, :, :] = gray_reshaped
            buffer_counter += 1
        #     print(curr_frame_nr)
        #     length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #     curr_frame_nr = cap.get(cv2.CAP_PROP_POS_FRAMES)
        #     # print("length", length, curr_frame_nr)
        #     # buffer.append(gray_reshaped)
        #     # print('running every', time.time() - ms_counter)
        # x = np.asarray(buffer)
        if buffer_counter + 1 > max_frames:
            buffer_counter = 0
            pred_ind = np.argmax(model(net_input).numpy()[0])
            pred_label = labels[pred_ind]
            # print("Making a prediction", pred_label)

            # for elem in buffer:
            # print(elem.dtype)
            # print('X_data shape:', np.array(buffer, dtype=np.float32).shape)
            # buffer_array = np.ndarray(buffer)
            # print()
        #     #     print(np.expand_dims(np.array(buffer), 0).shape)
        #     pred = "I dumb"
        #     print(pred)

            # x = np.expand_dims(np.array(buffer), axis=0).astype(np.float32)
            # # print(x.shape)
            # pred_ind = np.argmax(model(x).numpy()[0])
            #
            # print()
            # print(np.argmax(model(x).numpy()))
            #     # print(x.dtype)
            #     # pred = labels[np.argmax(model(x).numpy)]
            #     # print(model(x))
            #     # cv2.rectangle(gray, (0, 0), (100, 100), (255, 0, 0), 2)
            # if time.time() - t > 10:
            #     break
        # write the output frame to disk
        cv2.putText(frame, "Prediction: " + pred_label, (2, 25),
                    font, 0.3, (0, 255, 0), 1, cv2.LINE_AA)
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('./videos/pred_output_5.avi', fourcc, 25,
                                     (160, 120), True)
        writer.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()


# %%
