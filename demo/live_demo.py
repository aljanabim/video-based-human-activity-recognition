from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to trained serialized model")
ap.add_argument("-i", "--input", required=True,
	help="path to our input video")
ap.add_argument("-o", "--output", required=True,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=128,
	help="size of queue for averaging")


args = vars(ap.parse_args())
# load the trained modelfrom disk
print("[INFO] loading model ...")
# model = load_model(args["model"])
lb = ["first","second", "third","mustafa", "einar","mikel"]

# initialize the image mean for mean subtraction along with the
# predictions queue

Q = deque(maxlen=args["size"])


# initialize the video stream(input video or camera)
# vs = cv2.VideoCapture(args["input"]) #<--- input video
vs = cv2.VideoCapture(0) #<--- camera
writer = None #pointer to output video file
(W, H) = (None, None)
# loop over frames from the video file stream
while True:
	# read the next frame from the file
    (grabbed, frame) = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
    if not grabbed:
        break

    if (W is None) or (H is None):
        (H, W) = frame.shape[:2]
        print(H,W)
    
    #TODO: frame to blacknwhite if necesary
    #TODO: keep deque with frames
    # frame to be augmented for output    
    output = frame.copy()

    # preds = model.predict(np.expand_dims(frame, axis=0))[0]
    preds = np.random.rand(1,6)
    Q.append(preds)

    # perform prediction averaging over the current history of
    # previous predictions
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb[i]
    print(label)
    
    # draw the activity on the output frame
    text = f"activity: {label}"
    cv2.putText(output, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
	# check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
            (W, H), True)
    # write the output frame to disk
    writer.write(output)
	# show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(1) & 0xFF
	
    if key == ord("q"): # break if the `q` key is pressed
        break
    
    # break


