import cv2 as cv
import numpy as np
from scipy.ndimage import label
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from skimage.measure import regionprops, regionprops_table, moments_normalized,moments_central
from skimage.morphology import (erosion, dilation, closing, diameter_opening, skeletonize, thin, disk,)
import matplotlib as mpl
from copy import copy
from skimage.util import invert
import math
from scipy.ndimage import label
import string
mpl.rc('image', cmap='gray')

def imshow(img,title=None,cmap=None):
    plt.imshow(img)
    plt.axis('off')
    if cmap:
        plt.set_cmap(cmap)
    if title:
        plt.title(title)
    plt.show()

def canny(img,hthresh, sigma=.4):
    gaukern2D= cv.getGaussianKernel(3,sigma)
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
    return localmaximg
    #imshow(localmaximg,"canny edge with High/low thresholds: "+str(hthresh))

def show_labels(lbl,nlbl=None,cmap='rainbow'):
    cmap = copy(mpl.cm.get_cmap(cmap))
    cmap.set_under(color="black")
    plt.imshow(lbl,cmap=cmap,interpolation='none',vmin=1,vmax=nlbl)
    plt.axis('off')
    plt.show()   

#Part 1: Label by rows
def labelByRows(b,minsize=-1,se=np.ones((3,3))): #b = image, se = structure defines feature connections
    lbl,nlbl = label(b,se) #lbl= labeled image, nlbl= how many found objs
    # first filter out regions below the minimum area
    rgn = regionprops(lbl) #measures properties of labled image 
    area = np.array([cc.area for cc in rgn]) #area of region in that area marked by regionprops
    big = (area>minsize) #compare to min size of area we want, remove what is less than that (periods)
    newlbl = np.append(0,np.cumsum(big)*big)  #zero at beginning is for background component
    lbl = newlbl[lbl]
    rgn = regionprops(lbl)  # do this again since we've changed the labels
    #imshow(lbl)
    nlbl = newlbl.max()
    #show_labels(lbl)
    # now identify rows and sort regions left to right within rows
    b2 = dilation(b,np.ones((1,2*b.shape[1])))
    lbl2,nlbl2 = label(b2)
    cx = np.array([cc.centroid[1] for cc in rgn]) #center point of obj for each object we have noted
    relbl = np.zeros(nlbl+1,dtype=np.int32)  # +1 for background
    for i in range(1,nlbl2+1):
        rowlbl = np.unique(lbl[lbl2==i]) #labels each element starting at 1, but it does add a 0 at the 
        #beginning bc there is a background component
        rowlbl = rowlbl[rowlbl!=0]  # ignore the background, this removes that 0 from the begining of the row
        order = np.argsort(np.argsort(cx[rowlbl-1]))  # -1 because cx doesn't include background
        # double argsort tells us where each item moves to in the sorted version
        #print("order",order)
        relbl[rowlbl] = rowlbl[order]
        #print("relbl",relbl)
    lbl = relbl[lbl]
    #show_labels(lbl)
    return lbl,nlbl                 

atag = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
# ground truth text
text = """Pangram 
The quick brown fox jumps over a lazy dog.

The New Colossus
By Emma Lazarus
Not like the brazen giant of Greek fame,
With conquering limbs astride from land to land;
Here at our sea-washed, sunset gates shall stand
A mighty woman with a torch, whose flame
Is the imprisoned lightning, and her name
Mother of Exiles. From her beacon-hand
Glows world-wide welcome; her mild eyes command
The air-bridged harbor that twin cities frame.
“Keep, ancient lands, your storied pomp!” cries she
With silent lips. “Give me your tired, your poor,
Your huddled masses yearning to breathe free,
The wretched refuse of your teeming shore.
Send these, the homeless, tempest-tost to me,
I lift my lamp beside the golden door!”"""

#comment out the above text statement and use lines 133-138 if you want to try other texts and images.
# print("Input a text as a ground truth, ensure it matches your image. For our original text, input: (not a care in the world) ")
# textinput= input()
# if textinput == "(not a care in the world)":
#     text = """Pangram The quick brown fox jumps over a lazy dog. The New Colossus By Emma Lazarus Not like the brazen giant of Greek fame, With conquering limbs astride from land to land; Here at our sea-washed, sunset gates shall stand A mighty woman with a torch, whose flame Is the imprisoned lightning, and her name Mother of Exiles. From her beacon-hand Glows world-wide welcome; her mild eyes command The air-bridged harbor that twin cities frame. “Keep, ancient lands, your storied pomp!” cries she With silent lips. “Give me your tired, your poor, Your huddled masses yearning to breathe free, The wretched refuse of your teeming shore. Send these, the homeless, tempest-tost to me, I lift my lamp beside the golden door!”"""
# else:
#     text=str(textinput)
text= text.translate(str.maketrans("","", string.punctuation)) #still having issues removing quotes but, i guess it is what it is. 
letters = [c for c in list(text) if (ord(c)>=65 and ord(c)<=90) or (ord(c)>=97 and ord(c)<=122)]

def alphabet(filename, resizeBy, imgthreshold, labelthreshold): #'alpha.png', 3, 1, 100
    alpha = cv.imread(filename,0).astype(np.float32)/255.0
    alpha= cv.resize(alpha,(resizeBy*alpha.shape[1],resizeBy*alpha.shape[0])) 
    ab = alpha < imgthreshold 
    albl,nalbl = labelByRows(ab,labelthreshold)
    return(albl, nalbl, alpha) #returns labeled, number of labels, image

def textcon(filename, resizeBy, imgthreshold, labelthreshold): #'bold.png', 3, .75, 126
    textimg = cv.imread(filename,0).astype(np.float32)/255.0
    textimg= cv.resize(textimg,(resizeBy*textimg.shape[1],resizeBy*textimg.shape[0])) 
    tb = textimg < imgthreshold 
    tlbl,ntlbl = labelByRows(tb,labelthreshold)
    return(tlbl, ntlbl, textimg) #returns labeled, number of labels, image

albl, nalbl, alpha = alphabet('alpha.png', 3, 1, 100)
tlbl, ntlbl, textimg= textcon('bold.png', 3, .75, 126)

def getTemplates(numberlabels, labelimg, img, cannythresh=1):
    if numberlabels != 52:
        raise ValueError("Adjust label thresholds for alphabet. The current value of numberlabels is " + str(numberlabels)+ ". it should be 52.")
    templates={}
    for i in range(numberlabels):#
        rgn= regionprops(labelimg)
        bbox=np.array([cc.bbox for cc in rgn])
        letter=img[(bbox[i][0]-1):(bbox[i][2]+1),(bbox[i][1]-1):(bbox[i][3]+1)] 
        skel= canny(letter,cannythresh) 
        #imshow(skel)
        e = atag[i]
        templates.update({e:skel})
    return (templates)
templates=getTemplates(nalbl ,albl, alpha)

def plotHits(hi,hj,img):
    '''Plot the locations of template hits, given their coordinates and the underlying image'''
    plt.imshow(img)
    plt.axis('off')
    plt.plot(hj,hi,'r+')
    plt.show()

#for alphabet
def plottedalpha(alpha,templates,i):
    structure=templates.get(i).astype(int)
    onlya= erosion(invert(alpha),structure)  
    hi,hj=np.nonzero(onlya)
    plotHits(hi,hj,invert(alpha))
# plottedalpha(alpha,templates,"M")
# plottedalpha(alpha, templates, "t")
# plottedalpha(alpha,templates, "a")

#for text img
def plottedtextimg(textimg, templates,i):
    structure=templates.get(i).astype(int)
    onlya= erosion(invert(textimg),structure)  
    hi,hj=np.nonzero(onlya)
    plotHits(hi,hj,invert(textimg))
# plottedtextimg(textimg, templates,'a')
# plottedtextimg(textimg, templates,'q')
# plottedtextimg(textimg, templates,'C')

def footprintOffset(foot):
    '''Figures out the offset between the footprint center and an active point'''
    (fi,fj) = erosion(foot,foot).nonzero()
    (ai,aj) = foot.nonzero()
    return(ai[ai.size//2]-fi,aj[aj.size//2]-fj)

def bestmatch(labelimg,templatelist,numlabels,img):
    matches = -1*np.ones(shape=(numlabels,1),dtype=int)
    for i in templatelist:
        structure=templatelist.get(i).astype(int) 
        onlya= erosion(invert(img),structure)
        hi,hj=np.nonzero(onlya)
        fi,fj= footprintOffset(structure)
        for k in range(len(hi)):
            label=labelimg[hi[k]+fi,hj[k]+fj] 
            if matches[label-1]!= -1:
                value=list(templatelist.values())[int(matches[label-1])]
                #pixinmatch=value.shape[0] * value.shape[1]
                pixinmatch= cv.countNonZero(value) 
                newvalue = list(templatelist.values())[list(templatelist).index(i)]
                new=cv.countNonZero(newvalue)
                #new=newvalue.shape[0]*newvalue.shape[1]
                if new>pixinmatch: 
                    matches[label-1] = list(templatelist).index(i)
            else:
                matches[label-1] = list(templatelist).index(i) 
    return matches       
amatches= bestmatch(albl,templates,nalbl, alpha)
tmatches= bestmatch(tlbl,templates,ntlbl, textimg)

rgna = regionprops(albl)
cxa = np.array([cc.centroid for cc in rgna])
rgnt = regionprops(tlbl)
cxt = np.array([cc.centroid for cc in rgnt])

def visualize(match,loc,img):#loc=centroid
    '''Shows the matches for each component, the component centroids, and the original image'''
    plt.imshow(img)
    for i in range(match.shape[0]):
        if match[i]>=0:
            plt.text(loc[i,1]-6,loc[i,0]-5,atag[int(match[i])],color='r')
    plt.axis('off')
    plt.show()
visualize(amatches,cxa,alpha) 
visualize(tmatches,cxt,textimg) 

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def grade_transcription(s,gt): 
    '''measures the number of errors in a transcription'''
    s2 = s.replace(" ","")
    gt2 = gt.replace(" ","")
    d = levenshteinDistance(s2,gt2)
    print(d,"errors in",len(gt2),"characters:",(1-d/len(gt2))*100,"percent correct")

def grader(templates, amatches, tmatches, atag, letters):
    listmatcha=[]
    listmatcht=[]
    for i in amatches:
        if int(amatches[i])!= -1:    
            listmatcha.append(list(templates.keys())[int(i)])
    outstra="".join(listmatcha)

    for i in tmatches:
        if int(tmatches[i])!= -1:
            listmatcht.append(list(templates.keys())[int(i)])
    outstrt="".join(listmatcht)
    letters="".join(letters)
    grade_transcription(outstra,atag) #92.3 percent accuracy > 82.6 percent accuracy
    grade_transcription(outstrt,letters) #84.6 percent accuracy > 74.3 percent accuracy
grader(templates, amatches, tmatches, atag, letters)


# #Part 3: Stats component matching 
# # translation & scale invariance (not rotation invariance because some letters rotated are other letters)

# #normalized moments are translation and scale invariant (print them, split them and reaarange as a martrix) 
# def nnclass(astat,tstat):
#     '''returns the index of each t vector of the closest a vector'''
#     d = distance_matrix(astat,tstat)  # args are MxK and NxK
#     #print(d.shape)
#     #imshow(d)
#     return d.argmin(0).astype(np.int32)

# rpa = regionprops(albl)
# ca = np.array([cc.extent for cc in rpa])
# ca2= np.array([cc.feret_diameter_max for cc in rpa])
# ca3=np.array([cc.area for cc in rpa])
# ca4=np.array([cc.orientation for cc in rpa])
# ca5= np.array([cc.euler_number for cc in rpa])
# astats=np.stack((ca,ca2,ca3,ca4,ca5)).T

# rpt = regionprops(tlbl)
# ct = np.array([cc.area for cc in rpt])
# ct2= np.array([cc.feret_diameter_max for cc in rpt])
# ct3=np.array([cc.area for cc in rpt])
# ct4=np.array([cc.orientation for cc in rpt])
# ct5= np.array([cc.euler_number for cc in rpt])
# tstats=np.stack((ct,ct2,ct3,ct4,ct5)).T

# matchesyay=nnclass(astats,tstats)
# visualize(matchesyay,cxt,textimg)

# listmatch=[]
# for i in matchesyay:
#     listmatch.append(list(templates.keys())[int(i)])
# strt="".join(listmatch)
# letter="".join(letters)
# grade_transcription(strt,letter) #17.4 percent accuracy
# #this was as good as I could manage to get it with combos of properties. 

# #oher ways of doing htis that we talked about: 
#     #weighting different stats above. 

#     #taking vertical slices of letters, cumsum of pixels and then plot them as
#     # "image profiles" then find area between curve. im assuming that should the area between 
#     # curves < a threshold or = 0 is considered a match