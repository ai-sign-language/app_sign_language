
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import os

# Set the location and name of the cfg file
CONFIG = './yolo/yolo-sign.cfg'

# Set the location and name of the pre-trained weights file
WEIGHTS = './yolo/yolo-sign_2000.weights'

# Set the location and name of the COCO object classes file
CLASSES = './yolo/names'

# Load the network architecture

import os  
print(os.path.exists(CLASSES))
print(os.path.exists(CONFIG))
print(os.path.exists(WEIGHTS))

# read class names from text file
classes = None
with open(CLASSES, 'r') as f:
     classes = [line.strip() for line in f.readlines()]
        
scale = 0.00392
conf_threshold = 0.5
nms_threshold = 0.4

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))



# function to get the output layer names 
# in the architecture
def get_output_layers(net): 
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print (output_layers)
    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def processImage(image):

    Width = image.shape[1]
    Height = image.shape[0]

    # read pre-trained model and config file
    net = cv2.dnn.readNet(WEIGHTS, CONFIG)

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
            
    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
    
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    
    # display output image    
    #out_image_name = "object detection"+str(index)
    #cv2.imshow(out_image_name, image)
    # wait until any key is pressed
    #cv2.waitKey()
     # save output image to disk
    #cv2.imwrite("out/"+out_image_name+".jpg", image)




capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
while True:
    stime = time.time()
    ret, frame = capture.read()
    if ret:
        processImage(frame)
        cv2.imshow('frame', frame)
        #print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

