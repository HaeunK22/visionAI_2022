import time
import cv2
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from numpy import random
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_boxes
from utils.plots import Annotator
from utils.torch_utils import select_device

import argparse
parser = argparse.ArgumentParser(description='eye_detection')
parser.add_argument('--data', help='data file name')
args = parser.parse_args()

FILE = args.data
SOURCE = './data/'+FILE+'.jpg'
WEIGHTS = './eye_detection/model/yolo.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.25
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False


def detect():
    source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE

    # Initialize
    device = select_device(DEVICE)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print('device:', device)

    # Load model
    model = attempt_load(weights, device=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Load image
    img0 = cv2.imread(source)  # BGR
    assert img0 is not None, 'Image Not Found ' + source

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=AUGMENT)[0]
    print('pred shape:', pred.shape)

    # Apply NMS
    pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

    # Process detections
    det = pred[0]
    print('det shape:', det.shape)

    s = ''
    s += '%gx%g ' % img.shape[2:]  # print string
    annotator = Annotator(img0, line_width=3, example=str(names))
    list = []
    if len(det):
        # Rescale boxes from img_size to img0 size
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
        # Write results
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            annotator.box_label(xyxy, color=colors[int(cls)])
    # Stream results
    eyes = det.cpu().numpy()
    eyes = eyes[:, 0:4]
    return eyes

rw = 0.3
rh = 3
rh1 = 6
if __name__ == '__main__':
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        eyes = detect()
    source, weights, imgsz = SOURCE, WEIGHTS, IMG_SIZE
    eyesw = eyes[:, 2] - eyes[:, 0]
    eyesh = eyes[:, 3] - eyes[:, 1]
    eyes[:, 0] = eyes[:, 0] - rw * eyesw
    eyes[:, 2] = eyes[:, 2] + rw * eyesw
    eyes[:, 1] = eyes[:, 1] - rh1 * eyesh
    eyes[:, 3] = eyes[:, 3] + rh * eyesh
    img0 = cv2.imread(source)
    for i in range(len(eyes)):
        imgs = img0[eyes[i, 1].astype(np.int64):eyes[i, 3].astype(np.int64), eyes[i, 0].astype(np.int64):eyes[i, 2].astype(np.int64)]
        if not os.path.isdir('result/' + FILE):
            os.mkdir('../result/' + FILE)
        cv2.imwrite('../result/' + FILE + '/result'+str(i)+'.jpg', imgs)
    print(eyes)
