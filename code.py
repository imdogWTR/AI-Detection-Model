#!usr/bin/python3

#Jetson imports to use the custom model for helmet detection
import jetson_inference
import jetson_utils

import argparse



parser = argparse.ArgumentParser(description="Helmet Detection using a Custom Model")

parser.add_argument("input_filename", type=str, help="Path to the input image")
parser.add_argument("output_filename", type=str, help="Path to save the output image")
parser.add_argument("--overlay", type=str, default="box,labels", choices=["none", "box", "labels", "keypoints", "line"], help="Overlay type to apply on the image")

opt = parser.parse_args()



img = jetson_utils.loadImage(opt.input_filename)

net = jetson_inference.detectNet(
    network = "ssd-mobilenet-v1",
    model = "helmet-model/ssd-mobilenet.onnx",
    labels = "helmet-model/labels.txt",
    input_blob = "input_0",
    output_cvg = "scores",
    output_bbox = "boxes",
    threshold = 0.5
)
detections = net.Detect(img, overlay=opt.overlay)

jetson_utils.saveImage(opt.output_filename, img)
print(f"Detected {len(detections)} objects in the image.")
print(f"Successfully saved detections results to {opt.output_filename}")