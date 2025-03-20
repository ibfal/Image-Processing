import cv2 as cv
import numpy as np
from scipy.interpolate import RegularGridInterpolator 
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='gray')
import math

def imshow(img,cap=None):
    plt.imshow(img)
    plt.axis('off')
    if cap:
        plt.title(cap)
    plt.show()

#interpolation 
#Scale down to 1000 pixels, maintianing aspect ratio
room = cv.imread('RGBroom.webp',cv.IMREAD_COLOR)[:,:,[2,1,0]]/255.0
print("original shape ", room.shape) 
height,width = room.shape[0:2]
gcf = math.gcd(height,width)
ratio = (height/gcf)/(width/gcf)
print("aspect ratio ", ratio)
pixel1000 = cv.resize(room,(30,30),interpolation = cv.INTER_AREA)
print("new size (excluding border): ", pixel1000.shape)
heightn,widthn = pixel1000.shape[0:2]
pixel1000 = cv.copyMakeBorder(pixel1000,1,1,1,1,cv.BORDER_REFLECT)
imshow(pixel1000, "Scaled shown image 900 pixels (+ 124 border pixels)")
print("shape of new image (including border): ", pixel1000.shape)

#scale back up to original dim using interpolation

xgiven=np.array(range(widthn+2))
ygiven=np.array(range(heightn+2))

#1. nearest neighbor interpolation 
interp= RegularGridInterpolator((xgiven,ygiven),pixel1000, bounds_error=False, fill_value=0.0)
cx = np.array(range(2,widthn*100+2))/100
cy = np.array(range(2,heightn*100+2))/100
cxgrid,cygrid=np.meshgrid(cx,cy,indexing='xy')
nn_interp= interp((cygrid,cxgrid),'nearest')
imshow(nn_interp, "nearest neighbor interpolation")
print("shape after nearest neighbor: ", nn_interp.shape)

#2. bilinear interpolation 
bilin_interp= interp((cygrid,cxgrid))
imshow(bilin_interp, "bilinear interpolation")
print("shape after bilinear: ", bilin_interp.shape)

#crop for closer look
imshow(nn_interp[1300:1800,1300:1800], "nearest neighbor crop ")
imshow(bilin_interp[1300:1800,1300:1800], "bilinear crop: ")

#describe differences:
#The biggest difference between they two interpolation methods is that nearest neighbor results in a more noticeably
# pixilated image with less sharpness. In the cropped image for nn_interp, there is a distinct division between color points. 
# The squares are all evenly sized. but with the bilin_interp, there is a smooth transition between colors, and a slightly more
#distict shape of the image appears.

img= cv.imread('RGBdog.jpeg',cv.IMREAD_COLOR)[:,:,[2,1,0]]/255.0
img= cv.copyMakeBorder(img,1,1,1,1,cv.BORDER_REFLECT)
#Resampling filters

#tv interferance 

def tv_interfere(img, period=100, amp=(1/4)):
    img_row, img_col= img.shape[0:2]
    amp=period*amp
    px= np.array(range(2,img_col+2))
    py= np.array(range(2,img_row+2))
    pxg, pyg = np.meshgrid(px,py, indexing='xy')
    tx=pxg+(amp*np.sin((2*math.pi*pyg)/period))
    ty=pyg
    interpol = RegularGridInterpolator((py,px),img, bounds_error=False, fill_value=0.0) 
    output=interpol((ty,tx))
    imshow(output, "tv")
tv_interfere(img)

#pinched_face: 
def pinched(img,k=2):
    img_row, img_col= img.shape[0:2]
    px= np.linspace(-100,100,img_col)
    py= np.linspace(-100,100,img_row)
    pxg, pyg = np.meshgrid(px,py, indexing='xy')
    theta=np.arctan2(pyg,pxg)
    r = np.sqrt(pxg**2+pyg**2)
    r= r+(r**(1/k))
    tx= (r)*np.sin(theta)
    ty= (r)*np.cos(theta)
    interpol = RegularGridInterpolator((py,px),img, bounds_error=False, fill_value=0.0) 
    new=interpol((tx, ty))
    imshow(new, "pinched")
pinched(img)
    
def circles(img,period=10):
    img_row, img_col= img.shape[0:2]
    px= np.linspace(-100,100,img_col)
    py= np.linspace(-100,100,img_row)
    pxg, pyg = np.meshgrid(px,py, indexing='xy')
    theta=np.arctan2(pyg,pxg)
    r = np.sqrt(pxg**2+pyg**2)
    r=r+(np.sin(r))
    theta=theta+(np.sin(r)/period)
    tx= (r)*np.sin(theta)
    ty= (r)*np.cos(theta)
    interpol = RegularGridInterpolator((py,px),img, bounds_error=False,fill_value=0.0) 
    new=interpol((tx,ty))
    imshow(new, "circles")
circles(img)

def whirl(img):
    img_row, img_col= img.shape[0:2]
    px= np.linspace(-100,100,img_col)
    py= np.linspace(-100,100,img_row)
    pxg, pyg = np.meshgrid(px,py, indexing='xy')
    theta=np.arctan2(pyg,pxg)
    r = np.sqrt(pxg**2+pyg**2)
    r=r-np.log(r)
    theta= theta-(2*math.pi*(1-r/r.max()))/np.sqrt(r)
    tx= (r)*np.sin(theta)
    ty= (r)*np.cos(theta)
    interpol = RegularGridInterpolator((py,px),img, bounds_error=False,fill_value=0.0) 
    new=interpol((tx,ty))
    imshow(new, "whirl")
whirl(img)


def kaleidoscope(img):
    img_row, img_col= img.shape[0:2]
    px= np.linspace(-100,100,img_col)
    py= np.linspace(-100,100,img_row)
    pxg, pyg = np.meshgrid(px,py, indexing='xy')
    theta=np.arctan2(pyg,pxg)
    r = np.sqrt(pxg**2+pyg**2)
    r=r-(math.pi)/r
    theta=theta*math.pi*2
    tx= (r)*np.sin(theta)
    ty= (r)*np.cos(theta)
    interpol = RegularGridInterpolator((py,px),img, bounds_error=False,fill_value=0.0) 
    new=interpol((tx,ty))
    imshow(new,"kaleidoscope")
kaleidoscope(img)


#i know some of them dont work exactly as intended. I did try my best to get them there 
# but i think i'm not understanding what operations do what in order to change images. 
#I also am unsure about using a function like r=r+f*(1/4)or something like that. I know
#if i could get a more detailed explanation of these things that would be really helpfull. 
#I also attemtped the mirror one, I split the y's and changed them and then stacked them,
#which obviously altered each half appropriately but resulted in the image havinga T shape
# /upsidedown T. I designed the Kaleidoscope one just for kicks and giggles, its not perfect 
# but it looks fun on my images. 