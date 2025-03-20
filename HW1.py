import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('image', cmap='gray')
import rawpy
import PIL

def imshow(img):
    plt.imshow(img)
    plt.axis('off')
    plt.show()

#demosaic function:

def demosaic(RawIMG, raw):
    if RawIMG.raw_pattern[0,0] == 0 and RawIMG.raw_pattern[0,1] == 1 and RawIMG.raw_pattern[1,0] == 3 and RawIMG.raw_pattern[1,1] == 2:
        R= raw[0::2,0::2]
        G = raw[0::2,1::2]/2+ raw[1::2,0::2]/2 
        B = raw[1::2,1::2]
        return (R,G,B)
    if RawIMG.raw_pattern[0,0] == 3 and RawIMG.raw_pattern[0,1] == 2 and RawIMG.raw_pattern[1,0] == 0 and RawIMG.raw_pattern[1,1] == 1:
        R = raw[1::2,0::2]
        G = raw[0::2,0::2]/2 + raw[1::2,1::2]/2 
        B = raw[0::2,1::2]
        return (R,G,B)        

# Color Scaling
def scaling(RawIMG, r,g,b): #this is a more cool pic
    for i in range(3):
        hi=float(RawIMG.camera_white_level_per_channel[i])
        lo=float(RawIMG.black_level_per_channel[i])
        if i== 0:
            scale_r=((r.astype(np.float32)-lo)/(hi-lo))
            sw_R= (RawIMG.daylight_whitebalance[i]*scale_r).clip(0.0,1.0)
        if i==1:
            scale_g=((g.astype(np.float32)-lo)/(hi-lo))
            sw_G= (RawIMG.daylight_whitebalance[i]*scale_g).clip(0.0,1.0)
        if i== 2:
            scale_b=((b.astype(np.float32)-lo)/(hi-lo))
            sw_B= (RawIMG.daylight_whitebalance[i]*scale_b).clip(0.0,1.0)
    return(sw_R,sw_G,sw_B)    
   
def scaling_after(RawIMG, r,g,b): #returns a more warm pic
    for i in range(3):
        hi=float(RawIMG.camera_white_level_per_channel[i])
        lo=float(RawIMG.black_level_per_channel[i])
        if i== 0:
            br= (RawIMG.daylight_whitebalance[i]*r)
            sw_R=((br.astype(np.float32)-lo)/(hi-lo)).clip(0.0,1.0)
        if i==1:
            bg= (RawIMG.daylight_whitebalance[i]*g)
            sw_G=((bg.astype(np.float32)-lo)/(hi-lo)).clip(0.0,1.0)
        if i== 2:
            bb= (RawIMG.daylight_whitebalance[i]*b)
            sw_B=((bb.astype(np.float32)-lo)/(hi-lo)).clip(0.0,1.0)
    return(sw_R,sw_G,sw_B)  
     
def scaling_color_hi_lo(RawIMG,r,g,b): #returns a pink pic
    for i in range(3):
        hi= [r.max(), g.max(), b.max()]
        lo= [r.min(), g.min(), b.min()]
        if i== 0:
            scale_r=(r.astype(np.float32)-lo[i])/(hi[i]-lo[i])
            sw_R= (RawIMG.daylight_whitebalance[i]*scale_r).clip(0.0,1.0)
        if i==1:
            scale_g=(g.astype(np.float32)-lo[i])/(hi[i]-lo[i])
            sw_G= (RawIMG.daylight_whitebalance[i]*scale_g).clip(0.0,1.0)
        if i== 2:
            scale_b=(b.astype(np.float32)-lo[i])/(hi[i]-lo[i])
            sw_B= (RawIMG.daylight_whitebalance[i]*scale_b).clip(0.0,1.0)    
    return(sw_R,sw_G,sw_B)   


#Gamma correction

def gamma_correct_difficult(c):    #to do it the hard/slow way.
    for m in range(c.shape[0]):
        for n in range(c.shape[1]):
            for k in range(c.shape[2]):
                inten= c[m,n,k]
                if inten < 0.0031308:
                    inten= 12.92*inten
                    c[m,n,k]=inten
                else: 
                    inten= ((inten**(1/2.4))*1.055)-0.055
                    c[m,n,k]=inten
    return(c)

def gamma_correct(c,gamma=None):   
    if gamma==None:
        a=(np.multiply(c,12.92, where=(c<0.0031308))).clip(0.0,1.1)
        b=(np.power(c,(1/2.4), where=(c>=0.0031308))*1.055-0.055).clip(0.0,1.1)
        return b+a
    else:
        b=np.power(c,gamma).clip(0.0,1.1)
        return b


def calibrate(rawIMG,specifc_gamma=None): 
    Raw = rawpy.imread(rawIMG)
    Rawimg = Raw.raw_image.copy()
    R,G,B = demosaic(Raw,Rawimg) 
    Rscale, Gscale, Bscale = scaling(Raw,R,G,B) 
    stacked = np.dstack((Rscale,Gscale,Bscale))
    finishedIMG = gamma_correct(stacked, specifc_gamma)
    RGBimg= imshow(finishedIMG)
    return RGBimg

calibrate('cvan.NEF')

#Reflection:
#I feel extremely proud of myself for figuring things out about these functions on my own. 
#It was really difficult, not understanding what is really going on with the programming of these concepts. 
# I did understand the concepts and the math, but I really struggled putting this into code. 
#I spent a lot of time messing around with the arrays and the functions, I littered the program with print 
# statements and imshow(). I had to do a lot of extra research on the programming to better understand 
# the modules we were using and how everything fit together. Once I figureed it out it was easy to program 
# the functions, I was just really concerned about the order things needed to be in. I am confused about how 
# to program interpolation, so i chose to omit that. I wrote conditionals in for the cases with the backwards 
# patterns. I had to adjust the parameters of some functions because i needed to use additional things within
#them. I did not want to create global variables, so parameters were the solution. I wrote three differnt versions
# of the scaling method to account for the orders i did things, but i only used the cooler form of the image for my 
# purposes. For gamma correct, I wrote out the longer way using for loops to iterate over the whole array but it takes
# way longer and its just not optimal. Overall, it was stressful but productive, and im satisfied with my work on this. 