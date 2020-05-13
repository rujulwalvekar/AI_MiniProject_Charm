# from charmboard import deep_learning_object_detection
from imutils.video import VideoStream
from flask import Response,Flask,render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
CLASSES=["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS=np.random.uniform(0,255,size=(len(CLASSES),3))
print("[INFO] loading model...")
net=cv2.dnn.readNetFromCaffe("charmboard/MobileNetSSD_deploy.prototxt.txt","charmboard/MobileNetSSD_deploy.caffemodel")
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
print("[INFO] starting video stream...")
vs = VideoStream(src=1).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect():
    global vs,lock
    while True:
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

        with lock:
            outputFrame=frame.copy()

def generate():
    #grab global references to output frame and lock
    global outputFrame,lock

    while True:
        with lock:
            #check if output frame available else skip
            if outputFrame is None:
                continue
            #encode frame in JPEG
            (flag,encodedImage)=cv2.imencode(".jpg",outputFrame)
            #ensure frame succesfully encoded
            if not flag:
                continue
        #yield output frame in byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+bytearray(encodedImage)+b'\r\n')

@app.route("/video_feed")
def video_feed():
    #return response generated along with specific media
    #type (mime type)
    return Response(generate(),mimetype="multipart/x-mixed-replace;boundary=frame")

#check if main thread
if __name__=='__main__':
    #construct argument parser and parse command line args
    ap=argparse.ArgumentParser()
    ap.add_argument("-p","--prototxt",required=True,help="path to Caffe 'deploy' prototxt file",default="charmboard/MobileNetSSD_deploy.prototxt.txt")
    ap.add_argument("-m","--model",required=True,help="path to caffe pre-trained model",default="charmboard/MobileNetSSD_deploy.caffemodel")
    ap.add_argument("-c","--confidence",type=float,default=0.2,help="minimum probability to filter weak detections")
    args=vars(ap.parse_args())

    #start thread that will perform
    t=threading.Thread(target=detect)
    t.daemon=True
    t.start()

    #start flask app
    app.run(debug=True,threaded=True,use_reloader=False,host='0.0.0.0',port=80)
#release video stream pointer
vs.stop()
