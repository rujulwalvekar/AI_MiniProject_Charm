import numpy as np
import csv
#https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/

class Searcher:
    def __init__(self,indexPath):
        #store our index path
        self.indexPath=indexPath

    def search(self,queryFeatures,limit=10):
        #initialise dictionary of results
        results={}
        with open(self.indexPath) as f:
            #initialize CSV reader
            reader=csv.reader(f)

            #loop over the rows in the index
            for row in reader:
                #parse out the image ID and features, then compute the chi-squared distance between the features in our index and our query features
                features=[float(x) for x in row[1:]]
                d=self.chi2_distance(features,queryFeatures)

                results[row[0]]=d
            #close the reader
            f.close()
        #sort our results, so that the smaller distances (ie the more relevant images are the front of the list)
        results=sorted([(v,k) for (k,v) in results.items()])
        #return our (limited) results
        return results[:limit]

    def chi2_distance(self, histA,histB,eps=1e-10):
        # compute the chi-squared distance
        for(a,b) in zip(histA,histB):
            d=0.5*np.sum([((a-b)**2)/(a+b+eps) for (a,b) in zip(histA,histB)])
        #return
        return d
            
