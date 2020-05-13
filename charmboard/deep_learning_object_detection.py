#Caffe implementation of Google MobileNet SSD detection network, with pretrained weights on VOC0712 and mAP=0.727.
#https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with-deep-learning-and-opencv/

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

#construct argument parse
ap=argparse.ArgumentParser()
ap.add_argument("-p","--prototxt",required=True,help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m","--model",required=True,help="path to caffe pre-trained model")
ap.add_argument("-c","--confidence",type=float,default=0.2,help="minimum probability to filter weak detections")
args=vars(ap.parse_args())

#initialise list of class labels MobileNet SSD was trained to detect, then generate a set of bounding box colors for each class
CLASSES=["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS=np.random.uniform(0,255,size=(len(CLASSES),3))

#load serialized model from disk
print("[INFO] loading model...")
net=cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

#initialize video stream, allow camera sensor to warnup, and initialise FPS counter
print("[INFO] starting video stream...")
vs=VideoStream(src=0).start()
time.sleep(2.0)
fps=FPS().start()

#loop over frames from video stream
while True:
    #grab frame from threaded video stream and resize it to have maximum width of 400 pixels
    frame=vs.read()
    frame=imutils.resize(frame,width=400)

    #grab frame dimensions and convert it to a blob
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007843,(300,300),127.5)

    #pass blob through network and obtain detections and predictions
    net.setInput(blob)
    detections=net.forward()
    for i in np.arange(0,detections.shape[2]):
        #extract the confidence associated with prediction
        confidence=detections[0,0,i,2]

        #filter out weak detections by ensuring the confidence is greater than min confidence
        if confidence > args["confidence"]:
            #extract index of class label from detections, then compute (x,y) coordinates of the bounding box for the object
            idx=int(detections[0,0,i,1])
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")

            #display the prediction on the frame
            label="{}:{:.2f}%".format(CLASSES[idx],confidence*100)
            #print("[INFO] {}".format(label))
            cv2.rectangle(frame,(startX,startY),(endX,endY),COLORS[idx],2)
            y=startY-15 if startY - 15 > 15 else startY+15
            cv2.putText(frame,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)

    #show output frame
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF

    #if `q` key was pressed, break from loop
    if key==ord("q"):
        break

    #update FPS counter
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()





'''
#load input image and construct an input blob for the image by resizing to a fixed 300X300 pixels and then normalizing it (normalization done via authors of MobileNet SSD implementation)
image=cv2.imread(args["image"])
(h,w)=image.shape[:2]
blob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),0.007843,(300,300),127.5)

#pass blob through network and obtain detections and predictions
print("[INFO] computing object detections....")
net.setInput(blob)
detections=net.forward()

#loop over the detections
for i in np.arange(0,detections.shape[2]):
    #extract the confidence associated with prediction
    confidence=detections[0,0,i,2]

    #filter out weak detections by ensuring the confidence is greater than min confidence
    if confidence > args["confidence"]:
        #extract index of class label from detections, then compute (x,y) coordinates of the bounding box for the object
        idx=int(detections[0,0,i,1])
        box=detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX,startY,endX,endY)=box.astype("int")

        #display the prediction
        label="{}:{:.2f}%".format(CLASSES[idx],confidence*100)
        print("[INFO] {}".format(label))
        cv2.rectangle(image,(startX,startY),(endX,endY),COLORS[idx],2)
        y=startY-15 if startY - 15 > 15 else startY+15
        cv2.putText(image,label,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,COLORS[idx],2)

cv2.imshow("Output",image)
cv2.waitKey(0)
'''