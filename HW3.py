import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.ndimage import label
mpl.rc('image', cmap='gray')

def imshow(img,title=None):
    plt.figure(figsize=(10,6))
    plt.imshow(img)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

bee= cv.imread('Agora_I_5477_Rotation1_300dpi.png',0).astype(np.float32)/255.0
bee = cv.copyMakeBorder(bee,5,5,5,5,cv.BORDER_REFLECT)

"""dancer= cv.imread('dancer.jpg',0).astype(np.float32)/255.0
dancer = cv.copyMakeBorder(dancer,5,5,5,5,cv.BORDER_REFLECT)

book = cv.imread('book.jpg',0).astype(np.float32)/255.0
book = cv.copyMakeBorder(book,5,5,5,5,cv.BORDER_REFLECT)

horse = cv.imread('horse.jpg',0).astype(np.float32)/255.0
horse = cv.copyMakeBorder(horse,5,5,5,5,cv.BORDER_REFLECT)

scarf = cv.imread('scarf.jpeg',0).astype(np.float32)/255.0
scarf = cv.copyMakeBorder(scarf,5,5,5,5,cv.BORDER_REFLECT)

neon = cv.imread('neon.jpg',0).astype(np.float32)/255.0
neon = cv.copyMakeBorder(neon,5,5,5,5,cv.BORDER_REFLECT)"""

#i did not work with anyone on this program. 


def marr_hildreth(img,threshold): 
    sigma=1.4 #I like it better with smaller sigma(.4)
    gaukern1D= cv.getGaussianKernel(9,sigma)
    gaukern=np.matmul(gaukern1D,gaukern1D.T)
    gauimg = cv.filter2D(src=img,ddepth=-1,kernel=gaukern) 
    gauimg= (gauimg**2)+(gauimg**2)
    lapkern=np.array([[0,0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0,0],[1,1,1,-4,-4,-4,1,1,1],[1,1,1,-4,-4,-4,1,1,1],[1,1,1,-4,-4,-4,1,1,1],[0,0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,0,0]]) 
    lapped=cv.filter2D(src=gauimg,ddepth=-1,kernel=lapkern)
    edgeimg=np.zeros(lapped.shape,np.float32)
    for i in range(1,lapped.shape[0]-1):
        for j in range(1,lapped.shape[1]-1):
            neg_count=0
            pos_count=0
            if lapped[i][j-1]<0: 
                neg_count+=1
            if lapped[i][j-1]>0:
                pos_count+=1
            if lapped[i-1][j]<0:
                neg_count+=1
            if lapped[i-1][j]>0:
                pos_count+=1
            if neg_count>0 and pos_count>0:
                edgeimg[i,j]=1
                if gauimg[i,j]<=threshold: 
                    edgeimg[i,j]=0    
    imshow(edgeimg,"Marr-hildreth with threshold: "+str(threshold))
    
#marr_hildreth(bee,.17)
#marr_hildreth(book,.4)
#marr_hildreth(dancer,.25)


def canny(img,gaurad,hthresh,lthresh):
    sigma= 1.4 
    gaukern2D= cv.getGaussianKernel(gaurad,sigma)
    gaukern=np.matmul(gaukern2D,gaukern2D.T)
    gauimg = cv.filter2D(src=img,ddepth=-1,kernel=gaukern) 

    sobxkern= np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobykern= np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    sobximg=cv.filter2D(src=gauimg,ddepth=-1,kernel=sobxkern)
    sobyimg=cv.filter2D(src=gauimg,ddepth=-1,kernel=sobykern)

    gmag= (sobximg**2)+(sobyimg**2)
    angle=np.degrees(np.round(np.arctan2(sobyimg,sobximg)/(math.pi/4))*(math.pi/4))  
    localmaximg=np.zeros(img.shape,np.float32)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            pix1=255
            pix2=255
            if angle[i,j] < 0:
                angle[i,j]+=180
            if (0<=angle[i,j]<22.5) or (157.5<=angle[i,j]<=180):
                pix1=gmag[i,j+1]
                pix2=gmag[i,j-1]
            elif (22.5<=angle[i,j]<67.5):
                pix1=gmag[i+1,j-1]
                pix2=gmag[i-1,j+1]
            elif (67.5<= angle[i,j]<112.5):
                pix1=gmag[i+1,j]
                pix2=gmag[i-1,j]
            elif (112.5<= angle[i,j]<157.5):
                pix1=gmag[i-1,j-1]
                pix2=gmag[i+1,j+1]
            if (gmag[i,j]>=pix1)and(gmag[i,j]>=pix2):
                localmaximg[i,j]=gmag[i,j]
                if localmaximg[i,j]>= hthresh:
                    localmaximg[i,j]=1.0
            else:
                localmaximg[i,j]=0      
    labeled, numbers = label(localmaximg, structure=np.array([[1,1,1],[1,1,1],[1,1,1]]))
    for i in range(1,numbers):
        upper= max(localmaximg[labeled==i])
        if upper<hthresh:
            localmaximg[labeled==i]=0
    imshow(localmaximg,"canny edge with High/low thresholds: "+str(hthresh)+" and "+str(lthresh))

canny(bee,3,.5,.3) 
# canny(book,3,.05,.01)
# canny(horse,3,.15,.05)
#canny(scarf,3,.05,.01)
# canny(neon,3,.4,.1)
# canny(bee,3,.17,.03)


#Reflection: I really enjoyed this one! I felt like i better understood the assignment. I still 
# did a lot of research to figure out what stuff was and to make sure i understood what everything 
# did on its own, but it was easy to grasp once i understood the process. A few questions arose after 
#getting it to work, but with your help and some more researching I got it all sorted and the images 
# look really cool to see the edge lines.  

#I noticed that in order to get more lines to appear i had to get the highthreshold low enough. I 
# wanted the essence of the image to remain so i had to lower that. the low threshold helepd reduce noise 
# as far as i could tell, when i added that the little squares of "edges" around stronger edges went away 
# so that was interesting I wanted to make sigma an optional parameter because when i made it smaller or 
# larger that changed the thickness of the lines. I prefer the canny one more just because it gives a 
# clearer and cleaner image than the Marr-hilldreth. Marr-hilldreth feels more blocky and canny gives the 
# feel of a black and white sketch which is interesting! 

