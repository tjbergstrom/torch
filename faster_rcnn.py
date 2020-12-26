# faster_rcnn.py
# Using the Faster R-CNN pre trained model for object detection
#
# python3 faster_rcnn.py


import torchvision.transforms as T
import torchvision
import torch
from PIL import Image
import numpy as np
import cv2
import imutils
import time
from rcnn import RCNN


def predictions(img_path, threshold):
	img = Image.open(img_path)
	transform = T.Compose([T.ToTensor()])
	img = transform(img)
	print("Getting predictions...")
	start = time.time()
	preds = model([img])
	print(f"{time.time()-start:.3f} seconds")
	labels = [rnn.coco_classes[i] for i in list(preds[0]["labels"].numpy())]
	boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(preds[0]["boxes"].detach().numpy())]
	probs = list(preds[0]['scores'].detach().numpy())
	probs = [probs.index(i) for i in probs if i > threshold][-1]
	boxes = boxes[:probs + 1]
	labels = labels[:probs + 1]
	return boxes, labels


def detection(img_path, threshold=0.5):
	boxes, labels = predictions(img_path, threshold)
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	for i in range(len(boxes)):
		cv2.rectangle(img, boxes[i][0], boxes[i][1], (0, 255, 0), 3)
		cv2.putText(img,labels[i], boxes[i][0], 0, 3, (0, 255,0), 3)
	img = imutils.resize(img, width=500)
	cv2.imshow("", img)
	cv2.waitKey(0)


if __name__ == "__main__":
	rnn = RCNN()
	model = rnn.faster
	model.eval()
	detection('1.jpg', 0.8)



##
