import os
import time

import cv2
import numpy as np
import onnxruntime as ort
try:
    from .utils.box_utils_numpy import hard_nms
except:
    from utils.box_utils_numpy import hard_nms


import argparse
parser = argparse.ArgumentParser(description='detect_hand')
parser.add_argument('--video_path', default="0", type=str, help='video capture')
parser.add_argument('--label_path', default="models/hand-labels.txt", type=str, help='label path')
parser.add_argument('--module_path', default="models/onnx/handv0.onnx", type=str, help='module path')
parser.add_argument('--confidences_threshold', default=0.7, type=float, help='module path')
parser.add_argument('--iou_threshold', default=0.3, type=float, help='module path')

args = parser.parse_args()

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

class_names = [name.strip() for name in open(args.label_path).readlines()]

ort_session = ort.InferenceSession(args.module_path)
input_name = ort_session.get_inputs()[0].name

try:
    _cap = cv2.VideoCapture(int(args.video_path))
except Exception as e:
    _cap = cv2.VideoCapture(args.video_path)

while True:

    _ret, orig_image = _cap.read()
    if orig_image is None:
        print("end")
        break

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))
    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    time_time = time.time()
    confidences, boxes = ort_session.run(None, {input_name: image})
    print("cost time:{}".format(time.time() - time_time))
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, \
        args.confidences_threshold, args.iou_threshold)

    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = "{}: {:.2f}".format(class_names[labels[i]], probs[i])

        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label, (box[0] + 20, box[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow('image', orig_image)
    key = cv2.waitKey(1)
    if key == 27:
        break
