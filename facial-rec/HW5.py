import cv2 as cv
import numpy as np
from scipy.ndimage import label
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import matplotlib as mpl
from math import ceil

mpl.rc('image', cmap='gray')

def imshow(img,cmap=None):
    plt.imshow(img)
    plt.axis('off')
    if cmap:
        plt.set_cmap(cmap)
    plt.show()
    
robeson = cv.imread('robeson.jpg',0).astype(np.float32)/255.0
imshow(robeson)

#create integral image
def integral_image(img):
    return np.cumsum(np.cumsum(img, axis=0), axis=1)
iir = integral_image(robeson)

# return the f1 values for any image.
def f1_values(img,integral):
    r_4x12 = integral[4:,12:]+integral[:-4,:-12]-integral[4:,:-12]-integral[:-4,12:] 
    f1 = r_4x12[:-4,:]-r_4x12[4:,:]
    f1pad = np.pad(f1,((4,4),(6,6)))
    plt.figure
    plt.imshow(img)
    y,x = np.nonzero(f1pad>20)#;
    plt.plot(x,y,'r.')#;
    plt.axis('off')
    plt.show()
    return (f1pad)

f1values= f1_values(robeson,iir)

def f2_values(img, integral):
    r_4x4 = integral[4:,4:]+integral[:-4,:-4]-integral[:-4,4:]-integral[4:,:-4]
    f2= 2*r_4x4[:,4:-4]-r_4x4[:,:-8]-r_4x4[:,8:]
    f2pad=np.pad (f2,((2,2),(6,6)))
    plt.figure
    plt.imshow(img)
    y,x = np.nonzero(f2pad>20)#;
    plt.plot(x,y,'r.')#;
    plt.axis('off')
    plt.show()
    return (f2pad)

f2values=f2_values(robeson,iir)

def showFaces(img,x,y):
    '''Draws approximate boxes around detected face points.'''
    plt.figure
    plt.imshow(robeson)
    ax = plt.gca()
    for i in range(len(y)):
        r = mpl.patches.Rectangle((x[i]-12,y[i]-12),24,36,edgecolor='r',fill=False)
        ax.add_patch(r)
    #plt.plot(x,y,'c.')#;
    plt.axis('off')
    plt.show()
    
# assumes you have already defined f1p and f2p above.
y,x = np.nonzero(np.logical_and(f1values>20,np.roll(f2values,-4,0)>16))#;
showFaces(robeson,x,y)

# TODO: write a single function that takes an image and returns the y,x coordinates of the 
# faces detected using the method above.  Call it on the image scaled by different amounts:
# 1.25, 0.8, 0.64, etc. (use cv.resize).
def face_coord(img,resizeamt):
    img=cv.resize(img,None,fx=resizeamt,fy=resizeamt)
    integralimg= integral_image(img)
    f1value=f1_values(img,integralimg)
    f2value=f2_values(img,integralimg)
    y,x=np.nonzero(np.logical_and(f1value>20,np.roll(f2value,-4,0)>16))
    showFaces(img,x,y)
    return np.dstack([x,y])

print(face_coord(robeson, .64))
print(face_coord(robeson, .8))
print(face_coord(robeson, 1.25))
print(face_coord(robeson, 1.75))

#TODO: To complete the detection system, we therefore need to do a few things:

#Write a general rectangle-sum function that can compute the sums for rectangles of any size
def sum_any_size(img,xdim,ydim):
    integral=integral_image(img)
    block=integral[xdim:,ydim:]+integral[:-xdim,:-ydim]-integral[xdim:,:-ydim]-integral[:-xdim,ydim:]
    return(block)

#Write functions that can compute the f1 and f2 filters, given a scale. For example, at scale 
# 1.25 the f1 filter will use 5x15 boxes and the f2 filter will use 5x5 boxes.
def filters_scale(img, scale):
    f1xdim=int(4*scale)
    f1ydim=int(12*scale)
    f2dims=int(4*scale)
    block1=sum_any_size(img,f1xdim,f1ydim)
    f1 = block1[:-4,:]-block1[4:,:]
    block2=sum_any_size(img,f2dims,f2dims)
    f2= 2*block2[:,4:-4]-block2[:,:-8]-block2[:,8:]
    f1xpad= ceil((img.shape[0]-f1.shape[0])/2)
    f2xpad= ceil((img.shape[0]-f2.shape[0])/2)
    f1ypad= ceil((img.shape[1]-f1.shape[1])/2)
    f2ypad= ceil((img.shape[1]-f2.shape[1])/2)
    f1pad = np.pad(f1,((f1xpad,f1xpad),(f1ypad,f1ypad)))
    f2pad= np.pad(f2,((f2xpad,f2xpad),(f2ypad,f2ypad)))
    return f1pad, f2pad
f1v,f2v= filters_scale(robeson,1.25)

#Use the filter computations in a function that takes an image plus scale as input, and returns 
# the y and x for all faces detected at that scale
def face_cord_scale(img, scale):
    f1val,f2val= filters_scale(img,scale)
    y,x = np.nonzero(np.logical_and(f1val>(20*scale),np.roll(f2val,-4,0)>(16*scale)))
    return np.dstack([x,y])

#Write one more function that will call the one above in a loop, at different scales separated 
# by multiples of 1.25. It should return the scale and coordinates for each detection.
def face_loops(img):
    scale=1.25
    while scale <=10:
        print("coordinates for faces at scale " + str(scale))
        f1v,f2v= filters_scale(robeson, scale)
        y,x = np.nonzero(np.logical_and(f1v>(20),np.roll(f2v,-4,0)>(16)))
        showFaces(img,x,y)
        print(face_cord_scale(img,scale))
        scale=scale+1.25  
face_loops(robeson)

#reflection
# I found this project really interesting, but i definitely see the pitfalls of doing face detection this way. 
#By trying other iamges, the detector hardly ever picted up on faces that were not white and that is unsettling
# because i never really thought of these programs as being able to be racist in that sense and that was a really
#itnersting fact to come out of this. I also noticed that there were many cases where sharp contrast between black 
# and white or dark shadows in pictures were picked as "faces" and im curious to see how to fix that. It seems prety 
# inconsistent and that's a tad bit annoying. 


# I did not work with anyone. I used the Numpy documentation and scikit documentation along with the resources you provided
