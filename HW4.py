import cv2 as cv
import numpy as np
from scipy.ndimage import label
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from skimage.measure import regionprops, regionprops_table, moments_normalized,moments_central
from skimage.morphology import (erosion, dilation, closing, opening, skeletonize, thin, disk,)
import matplotlib as mpl
from copy import copy
from skimage.util import invert
mpl.rc('image', cmap='gray')

def imshow(img,title=None,cmap=None):
    plt.imshow(img)
    plt.axis('off')
    if cmap:
        plt.set_cmap(cmap)
    if title:
        plt.title(title)
    plt.show()

text = cv.imread('bold.png',0).astype(np.float32)/255.0
#imshow(text)
tb = text < 0.5  # binarized text

def show_labels(lbl,nlbl=None,cmap='rainbow'):
    cmap = copy(mpl.cm.get_cmap(cmap))
    cmap.set_under(color="black")
    plt.imshow(lbl,cmap=cmap,interpolation='none',vmin=1,vmax=nlbl)
    plt.axis('off')
    plt.show()
    
lbl,nlbl = label(tb,structure=np.ones((3,3)))

#Part 1: Label by rows
# understand purpose of this code answer questions below. Relies on morphology in some places

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
                  
tlbl,ntlbl = labelByRows(tb,14)
#show_labels(tlbl)

#TODO
#1. What is the purpose of the first call to regionprops?
        #it lists the properties of each labeled portion of an image. 

#2. Why does regionprops need to be called a second time?
        # we have adjusted the labels and removed what we dont want considered an element, 
        # so in order to make sure the properties are correct for all elements we want 
        # labeled we have to call it again for the new relabeled image

#3. What is the purpose of the dilation call? Explain the unusual structuring element.
        # the dilation shows us where the rows of text are. It tells us how much to dilate 
        # each element by to determine where there are rows and how far to extend it to.
       
#4. Why do we need the centroids of the components? idk about this??????
        # figure out proper order

#5. What does one iteration of the loop achieve? #or this?????
        # first line asigns a unique label to each letter of the line, it does include a 0 
        # for the background componenet at thee beginning of the line. The second line removes that 
        # 0 background component. then the double argsort tells us where each object moved to after 
        # sorting and the background isnt considered as a component. relbl then uses the new order as 
        # labels for the entire image.  
        # rowlbl=[0 1 2 3 4 5 6 7]
        # rowlbl= [1 2 3 4 5 6 7]
        # order = [0 1 2 3 4 5 6]
        # relbl= [0 1 2 3 4 5 6 7 0 0 0 0 0 0 .....] why is there a 7?????
        # each loop adds new labels to the end of relbl.

#6. What are the contents of relbl as computed within the loop? 
    #relbl consists of new labels 0-580 (but they arent in order on every line????)
#7. What does the line lbl = relbl[lbl] achieve?
    #reassigns the labels of the image to be what was ordered, sorted, and relabled


#Part 2: template matching
# template image
alpha = cv.imread('alpha.png',0).astype(np.float32)/255.0
# imshow(alpha)
ab = alpha < 0.5
albl,nalbl = labelByRows(ab,14)
# show_labels(albl)
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
#print(len(text))
letters = [c for c in list(text) if (ord(c)>=65 and ord(c)<=90) or (ord(c)>=97 and ord(c)<=122)]
#print(len(letters))
#print(letters)

textimg = cv.imread('bold.png',0).astype(np.float32)/255.0
#imshow(text)
tb = textimg < 0.5  # binarized text


templates={}
for i in range(nalbl):
    rgn= regionprops(albl)
    bbox=np.array([cc.bbox for cc in rgn])
    letter=alpha[bbox[i][0]:bbox[i][2],bbox[i][1]:bbox[i][3]]
    letter = invert(letter)
    #imshow(letter)
    #a= dilation(a) if use skelteton, do we have to use skeleton?
    skel=thin(letter) #could remove opening, but less matches
    #imshow(skel)
    e = atag[i]
    templates.update({e:skel})
# imshow(1-templates.get('T'), "template for T")
# imshow(1-templates.get('q'), "template for q")
# imshow(1-templates.get('a'), "template for a")
# imshow(1-templates.get('Z'), "template for Z")

def plotHits(hi,hj,img):
    '''Plot the locations of template hits, given their coordinates and the underlying image'''
    plt.imshow(img)
    plt.axis('off')
    plt.plot(hj,hi,'r+')
    plt.show()

#combined eroding and plotting functions 
#for alphabet
def plottedalpha(alpha,templates,i):
    structure=templates.get(i).astype(int)
    onlya= erosion(invert(alpha),structure)  
    hi,hj=np.nonzero(onlya)
    plotHits(hi,hj,invert(alpha))
# plottedalpha(alpha,templates,"M")
# plottedalpha(alpha, templates, "t")
# plottedalpha(alpha,templates, "o")

#for text img
def plottedtextimg(textimg, templates,i):
    structure=templates.get(i).astype(int)
    onlya= erosion(invert(textimg),structure)  
    hi,hj=np.nonzero(onlya)
    plotHits(hi,hj,invert(textimg))
# plottedtextimg(textimg, templates,'c')
# plottedtextimg(textimg, templates,'q')
# plottedtextimg(textimg, templates,'I')



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
                pixinmatch=value.shape[0] * value.shape[1]
                newvalue = list(templatelist.values())[list(templatelist).index(i)]
                new=newvalue.shape[0]*newvalue.shape[1]
                if new>=pixinmatch: #more matches are correct than if its strictly >
                    matches[label-1] = list(templatelist).index(i)
            else:
                matches[label-1] = list(templatelist).index(i) 
    return matches       

amatches= bestmatch(albl,templates,nalbl, alpha)
tmatches= bestmatch(tlbl,templates,ntlbl, textimg)

def only1match(labelimg,templatelist,numlabels): #this is extra and i dont want to delate it because it is still cool.
    best=np.empty(shape=(numlabels,1),dtype=str) #2 if n shows up
    rgn=regionprops(labelimg)
    bbox=np.array([cc.bbox for cc in rgn])
    for n in range(numlabels): #numlabels
        #print(n)
        matches=[]
        countmatch=0
        component=alpha[bbox[n][0]:bbox[n][2],bbox[n][1]:bbox[n][3]] #labelimg
        #best[n][0]=n #full n number not showing up
        #print(component.shape[0]*component.shape[1])
        for i in templatelist:
            structure=templatelist.get(i).astype(int)
            onlya= erosion(invert(component),structure) 
            if len(np.nonzero(onlya)[0])!=0 and len(np.nonzero(onlya)[1]) !=0:
                matches.append(i)
                countmatch+=1  
        #print(matches) 
        if len(matches)>=2:
            shapes=[]
            for m in matches:
                shapes.append(templatelist.get(m).shape)
            for s in shapes:
                a = s[0]*s[1]
                if a>(component.shape[0]*component.shape[1]):
                    a=0
                shapes[shapes.index(s)]=a
            #print(shapes)
            #shapes.reverse()
            maxindex= shapes.index(max(shapes))
            #index=len(shapes)-maxindex -1 
            best[n]= matches[maxindex]
        elif len(matches)==1:
            best[n]= matches[0]
        elif len(matches)==0:
            print("here")
            best[n]= "None"
    print("this is best: ")
    return best


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
# visualize(amatches,cxa,alpha)
# visualize(tmatches,cxt,textimg)



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

def grade_transcription(s,gt): #our string join() atag and the ground truth
    '''measures the number of errors in a transcription'''
    s2 = s.replace(" ","")
    gt2 = gt.replace(" ","")
    d = levenshteinDistance(s2,gt2)
    print(d,"errors in",len(gt2),"characters:",(1-d/len(gt2))*100,"percent correct")
    
# TODO:  Apply the function above to your results.
listmatcha=[]
for i in amatches:
    if int(amatches[i])!= -1:    
        listmatcha.append(list(templates.keys())[int(i)])
outstra="".join(listmatcha)
listmatcht=[]
for i in tmatches:
    if int(tmatches[i])!= -1:
        listmatcht.append(list(templates.keys())[int(i)])
outstrt="".join(listmatcht)
letters="".join(letters)
grade_transcription(outstra,atag) #82.6 percent accuracy
grade_transcription(outstrt,letters) #74.3 percent accuracy 


#Part 3: Stats component matching 
# what stats? look at regionprops (scikit), want nondepenedent measurements= translatio &scale
#invariance (not rotation invariance because some letters rotated are other letters)

#normalized moments are translation and scale invariant (couldnt figure out how to make it work with the dimensions) 

def nnclass(astat,tstat):
    '''returns the index of each t vector of the closest a vector'''
    d = distance_matrix(astat,tstat)  # args are MxK and NxK
    #print(d.shape)
    #imshow(d)
    return d.argmin(0).astype(np.int32)

rpa = regionprops(albl)
ca = np.array([cc.euler_number for cc in rpa])
ca2= np.array([cc.feret_diameter_max for cc in rpa])
ca3=np.array([cc.perimeter_crofton for cc in rpa])
astats=np.stack((ca,ca2,ca3)).T

rpt = regionprops(tlbl)
ct = np.array([cc.euler_number for cc in rpt])
ct2= np.array([cc.feret_diameter_max for cc in rpt])
ct3=np.array([cc.perimeter_crofton for cc in rpt])
tstats=np.stack((ct,ct2,ct3)).T

matchesyay=nnclass(astats,tstats)
visualize(matchesyay,cxt,textimg)

listmatch=[]
for i in matchesyay:
    listmatch.append(list(templates.keys())[int(i)])
strt="".join(listmatch)
letter="".join(letters)
grade_transcription(strt,letter) #31.4 percent accuracy
#this was as good as I could manage to get it with combos of properties. 

#REFLECTION:
#I really enjoyed this project. I liked making templates and being able to see how the operations changed the appearance
#of the template. I felt like part 1 and part 3 were the most challenging, just because normally i like to program things
#myself in order to figure out what they do and with part 1 i couldnt necesarily do that. And for part three, I was able to 
# use statistics, but it wasnt necessarily the ones i wanted. I couldnt get the statistics where i wanted grade wise, but I 
# tried my best on that but was not necessary sucessful in implimenting what i wanted. I feel like i learned a lot about different 
# ways to approach these questions and that even though my first implimentation wasnt exactly what we were going for it still did 
# work well. I also got creative with my options because there were less isntructions on types of things we needed to use like 
# dictionaries instead of lists. I feel really confident in my solutions for this project and im proud of myself for being able to 
#think outside the box when needed. 

#I think the statistics probably SHOULD have worked better, because by logical assumptions letters that look the same should be in 
# a similar area of a plane, but there is still confusion of letters like c,b,p,q that have similar structures and that might result
# in certain neighborhods overlapping and matches getting convoluted. I think the tempalte matching for templates and weeding out multple 
# matches was a bit more practical since it allowed you to be specific in what the difference between those templates were(in this case it 
# was their size). It yeilded a more accruate response. 

#i feel like i and j had no matches because their dots were missing. I wonder if there is a way to take the image, separte the text into a list 
# and then remove characters that are more specifcally punctuation. instead of all dots and colons etc. I have done this before on text files and 
#am curious if this will work on images and then yeild more accurate responses for the matching. 

# I did use numpy, scipy, and scikit documentation on fucntions to see what parameters and other things were needed for functions 
# and what they would return. I did bounce some ideas off of emi when i had some weird things happening but i ended up solving my 
#own problem by debugging my code piece by piece after thinking through the process a little longer. 
