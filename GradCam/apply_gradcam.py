# import the necessary packages
from GradCam.gradcam import GradCAM
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2

import sys
sys.path.append("/home/maxdox/Desktop/DeepLearning/Project/video-based-human-activity-recognition/")
from training import inception_tuning



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg",
	choices=("vgg", "resnet","inception", "finetuned"),
	help="model to be used")
args = vars(ap.parse_args())


# initialize the model to be VGG16

# check to see if we are using ResNet
IMSIZE = 224
if args["model"] == "inception":
	Model = InceptionV3
elif args["model"] == "resnet":
	Model = ResNet50
elif args["model"] == "vgg":
    Model = VGG16
if args["model"] == "finetuned":
	model = inception_tuning.load_model(True,"/home/maxdox/Desktop/DeepLearning/Project/video-based-human-activity-recognition/")
	IMSIZE = 160
	LABELS = ["boxing","handclapping", "handwaving","jogging", "running","walking"]
else:
		
	# load the pre-trained CNN from disk
	print("[INFO] loading model...")
	model = Model(weights="imagenet")

# load the original image from disk (in OpenCV format) and then
# resize the image to its target dimensions

orig = cv2.imread(args["image"])
resized = cv2.resize(orig, (IMSIZE, IMSIZE))
# load the input image from disk (in Keras/TensorFlow format) and
# preprocess it
image = load_img(args["image"], target_size=(IMSIZE, IMSIZE))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = imagenet_utils.preprocess_input(image)


# use the network to make predictions on the input image and find
# the class label index with the largest corresponding probability
preds = model.predict(image)
i = np.argmax(preds[0])
# decode the ImageNet predictions to obtain the human-readable label
if args["model"] == "finetuned":
	label = LABELS[i]
	prob = 0.5
else:
	decoded = imagenet_utils.decode_predictions(preds)
	(imagenetID, label, prob) = decoded[0][0]

label = "{}: {:.2f}%".format(label, prob * 100)
print("[INFO] {}".format(label))

# initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)
# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

# draw the predicted label on the output image
cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
	0.5, (255, 255, 255), 2)
# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([orig, heatmap, output])
output = imutils.resize(output, height=700)
cv2.imshow("Output", output)
cv2.waitKey(0)
