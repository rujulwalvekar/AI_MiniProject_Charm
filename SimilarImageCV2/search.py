from colordescriptor import ColorDescriptor
from searcher import Searcher
import argparse
import cv2

#construct argument parser and parse arguments
ap=argparse.ArgumentParser()
ap.add_argument("-i","--index",required=True,help="Path to where computed index will be stored")
ap.add_argument("-q","--query",required=True,help="Path to query image")
ap.add_argument("-r","--result-path",required=True,help="Path to the result path")
args=vars(ap.parse_args())

#initialise image descriptor
cd=ColorDescriptor((8,12,3))

#load query and describe it
query=cv2.imread(args["query"])
querysmol=cv2.resize(query,(960,540))
features=cd.describe(query)

#perform search
searcher=Searcher(args["index"])
results=searcher.search(features)
print(results)
#display query
cv2.imshow("Query",querysmol)

#loop over rsults
for(score,resultID) in results:
    #load result image and display it
    result=cv2.imread(args["result_path"]+"/"+resultID)
    resultsmol=cv2.resize(result,(960,540))
    cv2.imshow("Result",resultsmol)
    cv2.waitKey(0)

'''
python3 index.py --dataset imagedata --index index.csv
python3 search.py --index index.csv --query queries/q1.jpg --result-path dataset
'''