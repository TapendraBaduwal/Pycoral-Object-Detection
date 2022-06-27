# based on https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py
#https://github.com/TannerGilbert/Google-Coral-Edge-TPU/blob/master/pycoral_object_detection.py
from imutils.video import VideoStream, FPS
import argparse
import time
import cv2
from PIL import Image, ImageDraw
import numpy as np
import os
import pathlib
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

class Detection:
    
    def __init__(self):
        self.models_path = "models_path"
        self.model_file = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
        self.label_file = "coco_labels.txt"
        
    # def draw_objects(self,image, objs, labels):
    #     draw = ImageDraw.Draw(image)
    #     for obj in objs:
    #         bbox = obj.bbox
    #         draw.rectangle(
    #             [(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='red')
    #         draw.text((bbox.xmin + 10, bbox.ymin + 10), '%s\n%.2f' %
    #                 (labels.get(obj.id, obj.id), obj.score), fill='red')
    #     displayImage = np.asarray(image)
    #     cv2.imshow('Coral Live Object Detection', displayImage)

    def detection(self):

        script_dir = pathlib.Path(__file__).parent.absolute()
        models_dir= os.path.join(script_dir, self.models_path)
        model_file = os.path.join(models_dir, self.model_file)
        label_file = os.path.join(models_dir, self.label_file)
        
        labels = read_label_file(label_file) if label_file else {}
        interpreter = make_interpreter(model_file)
        interpreter.allocate_tensors()

        # Initialize video stream
        cam = cv2.VideoCapture(0) #6 for realsense RGB
        
        while True:
            try:
                ret, frame = cam.read()
                orig = frame.copy()
                image = Image.fromarray(frame)
                _, scale = common.set_resized_input(
                    interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
                interpreter.invoke()
                objs = detect.get_objects(interpreter, 0.4, scale)
                #print(objs)
                #self.draw_objects(image, objs, labels)
                objs_results = []
                for obj in objs:    
                    box = obj.bbox
                    startX = box.xmin
                    startY = box.ymin
                    endX = box.xmax
                    endY = box.ymax

                    (startX, startY, endX, endY) = box
                    label = labels[obj.id]
                    cv2.rectangle(orig, (startX, startY), (endX, endY),(0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    text = "{}: {:.2f}%".format(label, obj.score * 100)
                    cv2.putText(orig, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    objs_results.append([label, box])
                    print(objs_results)
                #return objs_results, orig
                cv2.imshow('Live Object Detection', orig)
                cv2.waitKey(10)

                if(cv2.waitKey(5) & 0xFF == ord('q')):
                    break
        
            except KeyboardInterrupt:
                break

        cv2.destroyAllWindows()

if __name__ == '__main__':
    obj = Detection()
    obj.detection()
    #obj.draw_objects()
    
    
    