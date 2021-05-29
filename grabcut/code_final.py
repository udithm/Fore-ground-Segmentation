import numpy as np
import argparse
import time
import cv2
import os

images = ["Image1.jpg","Image2.jpg","Image3.jpg"]
rec = [[20,20,300,299],[15,15,400,220],[80,30,290,150]]

for i in [0,1,2]:
    #print("Hello")
    #print(images[i])
    #print(rec[i])
    image = cv2.imread(images[i])
    mask = np.zeros(image.shape[:2], dtype="uint8")

    fgModel = np.zeros((1,65), dtype="float")
    bgModel = np.zeros((1,65), dtype="float")

    start = time.time()
    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rec[i], bgModel,fgModel, iterCount=100, mode=cv2.GC_INIT_WITH_RECT)
    end = time.time()
    print("Image",i)
    print("Applying GrabCut took {:.2f} seconds".format(end - start))

    values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
    )

    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),0, 1)

    outputMask = (outputMask * 255).astype("uint8")
    output = cv2.bitwise_and(image, image, mask=outputMask)
    
    name = "outputMask "+str(i)+" Runtime: "+str(end-start)
    cv2.imshow(name,outputMask)
    k = cv2.waitKey(0)
    if k == ord('s'): # wait for 's' key to save and exit
        save_name = "Image"+str(i+1)+"_seg.png"
        cv2.imwrite(save_name,outputMask)
        cv2.destroyAllWindows()
    #Press Any key to get next image

    