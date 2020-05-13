""" USAGE
python3 live_demo.py -o here.avi -m ../models/trained_models/LSTM_50epochs_trimmed/ \
    -i ../data/kth-actions/video/handwaving/person03_handwaving_d1_uncomp.avi
"""


from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import argparse
import pickle
import cv2
import time


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained serialized model")
ap.add_argument("-i", "--input", default=None,
	help="path to our input video")
ap.add_argument("-o", "--output", default=None,
	help="path to our output video")
ap.add_argument("-s", "--size", type=int, default=6,
	help="size of queue for averaging")
ap.add_argument("-bs", "--buffer_size", type=int, default=16,
	help="size of queue for averaging")


def main():
    args = vars(ap.parse_args())

    print("[INFO] loading model ...")
    model = load_model(args["model"])
    lb = ["boxing","handclapping", "handwaving","jogging", "running","walking"]

    Buffer = deque(maxlen=args["buffer_size"]) #frame buffer
    Q = deque(maxlen=args["size"]) #prediction buffer


    # initialize the video stream(input video or camera)
    if args["input"] is not None:
        vs = cv2.VideoCapture(args["input"])
    else:
        vs = cv2.VideoCapture(0) #<--- camera

    label = ""


    writer = None #pointer to output video file
    (W, H) = (None, None)

    counter =0
    while True:

        counter =np.mod(counter+1,1999)

        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        if (W is None) or (H is None):
            (H, W) = frame.shape[:2]
            print(H,W)
        
        # frame to be augmented for output
        model_input = cv2.resize(frame,(160,160))
        model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]
        Buffer.append(cv2.resize(model_input,(160,160)))


        output = frame.copy()
        if len(Buffer)==16 and np.mod(counter, 5):
            preds = model(np.repeat(np.expand_dims(np.array(Buffer), axis=[0,4]), repeats=3, axis=4  )).numpy()
            Q.append(preds)
            # perform prediction averaging 
            results = np.array(Q).mean(axis=0)
            i = np.argmax(results)
            label = lb[i]
            print(label)
        
        # draw the activity on the output frame
        text = 'activity: {}'.format(label)
        cv2.putText(output, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # show the output image
        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF
        

        # write the output frames to disk
        if writer is None and args["output"] is not None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                (W, H), True)
        else:
            time.sleep(0.1)
        writer.write(output)

        # break if the `q` key is pressed-*
        if key == ord("q"): 
            break


if __name__ == "__main__":
    main()