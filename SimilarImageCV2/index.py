from colordescriptor import ColorDescriptor
import argparse
import glob
import cv2

#construct argument parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="Path of folder where images to be indexed")

ap.add_argument("-i","--index",required=True,help="Path to where computed index be stored")
args=vars(ap.parse_args())

#initialise color descriptor
cd=ColorDescriptor((8,12,3))

#open output index file for writing
output=open(args["index"],"w")

#use glob to grab the image paths and loop over them
for imagePath in glob.glob(args['dataset']+"/*.jpg"):
    #extract the image ID (ie the unique filename) from the image path and load the image itself
    imageID=imagePath[imagePath.rfind("/")+1:]
    image=cv2.imread(imagePath)

    #describe the image
    features=cd.describe(image)
    features=[str(f) for f in features]
    output.write('%s,%s\n' % (imageID,",".join(features)))

#close the index file
output.close()