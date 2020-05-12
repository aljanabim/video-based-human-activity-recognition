import cv2
import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# model = tf.keras.models.load_model('./training/trained_models/LSTM_70epochs')

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
height = frame.shape[0]
width = frame.shape[1]
font = cv2.FONT_HERSHEY_SIMPLEX
print(ret, frame.shape)
dummy_data = np.random.random((16, width, height, 3))
print(dummy_data.shape)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.rectangle(gray, (0, 0), (100, 100), (255, 0, 0), 2)
    cv2.putText(frame, "Prediction: ", (10, height - 25),
                font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
