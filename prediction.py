""" Object detection macro are included in this module """

import cv2
import pickle
from PIL import Image
import time
from .log import get_console_logger
import imutils
from pycoral.utils.dataset import read_label_file
from pycoral.adapters import common,  detect
from pycoral.utils.edgetpu import make_interpreter
from cv_bridge import CvBridge
import rospy
# from sensor_msgs.msg import Image as im

confidence = None
model = None
labels = None

console = get_console_logger("prediction")
# viz_publisher = rospy.Publisher('/image_with_detections', im, queue_size=5)


def setup_model(model_path, labels_path, logger=None):
    if not logger:
        console.warning("Logger is not set, using the console")
        logger = console
    global confidence, model, labels, interpreter
    confidence = 0.3
    labels = {}
    # loop over the class labels file
    while True:
        try:
            labels = read_label_file(labels_path) if labels_path else {}
            interpreter = make_interpreter(model_path)
            interpreter.allocate_tensors()
            logger.info("TPU detected")
            break
        except Exception as e:
            logger.error(e)
            console.info("Will try connecting to USB-TPU every 5 second")
            time.sleep(5)


def _predict_pycoral(rgb_image,log_file):
    '''Prediction of the object using image-net model and publishe data to a ros topic'''
    global model, interpreter
    try:
        # t1 = time.time()
        frame = imutils.resize(rgb_image, width=500)
        orig = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        _, scale = common.set_resized_input(interpreter, frame.size, lambda size: frame.resize(size, Image.ANTIALIAS))
        interpreter.invoke()
        results = detect.get_objects(interpreter, 0.4, scale)
        # print(results)
        label_results = []

        for obj in results:    
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
            label_results.append([label, box])
        # cv2.imshow('Live Object Detection', orig)
        # cv2.waitKey(10)
        bridge = CvBridge()
        ros_img = bridge.cv2_to_imgmsg(orig,"bgr8")
        # viz_publisher.publish(ros_img)
        return label_results, ros_img
    except Exception as e:
        log_file.error(e)


def predict(image,log_file):
    """ Implements object detection """
    detections, ros_img = _predict_pycoral(image,log_file)
    return detections, ros_img


if __name__ == "__main__":
    setup_model("g", "f")