


import torchvision



class RCNN:
	def __init__(self):
		self.faster = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
		self.mask = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
		self.coco_classes = [
			'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
			'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
			'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
			'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
			'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
			'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
			'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
			'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
			'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
			'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
			'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
			'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
		]


	def color_masks(img):
		colors = [
		    [0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255],
		    [80, 70, 180], [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]
		]
		r = np.zeros_like(img).astype(np.uint8)
		g = np.zeros_like(img).astype(np.uint8)
		b = np.zeros_like(img).astype(np.uint8)
		r[img == 1], g[img == 1], b[img == 1] = colors[random.randrange(0, len(colors) - 1)]
		color_mask = np.stack([r, g, b], axis=2)
		return color_mask
