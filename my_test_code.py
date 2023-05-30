import cv2
import yolov5
import numpy
from yolov5.models.yolo import SegmentationModel

# model = yolov5.load('yolov5s.pt')
model = yolov5.load('yolov5s-seg.pt')
# model = SegmentationModel(yolov5.load('yolov5s-seg.pt'), cfg="yolov5s-seg.yaml")
# model = yolov5.load("best.pt")
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.max_det = 1000  # maximum number of detections per image

# perform inference
im = cv2.imread("bus.jpg")
results = model(im)

# im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# results = model(im_rgb)

# parse results
predictions = results.pred[0]
boxes = predictions[:, :4]  # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]
print(scores)
results.save(save_dir='results/')
