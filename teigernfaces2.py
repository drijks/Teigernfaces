import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

#takes an image and returns a list of each pixel's RGB values and xy coordinates
def getdata(img1):
    width, height = img1.size
    imgdata = []
    for x in range(width):
        for y in range(height):
            pixx = img1.getpixel((x,y))
            imgdata.append((pixx[0],pixx[1],pixx[2],x,y))
    return imgdata

#splits the image data into two lists with rgb values listed individually instead of as tuples - the xy coordinates are not currently in use but this may be incorporated in future versions
def tomatlist(m):
    matlist = []
    matcoord = []
    for i in range(len(m)):
        matlist.append(m[i][0]) #red value
        matlist.append(m[i][1]) #green value
        matlist.append(m[i][2]) #blue value
        matcoord.append(m[i][3]) #x coordinate
        matcoord.append(m[i][4]) #y coordinate
        matcoord.append(0) #this is only to keep the two lists at the same length
    return [matlist,matcoord]

#this does the same as the above function but also converts the images to grayscale
def tomatlistgscale(m):
    matlist = []
    matcoord = []
    for i in range(len(m)):
        matlist.append(0.2126*m[i][0]/256 + 0.7152*m[i][1]/256 + 0.0722*m[i][2]/256)
        matcoord.append(m[i][3])
        matcoord.append(m[i][4])
        matcoord.append(0)
    return [matlist,matcoord]

#Don't think I actually used this in this version, it was used for testing earlier on and will probably get deleted eventually
def matSubtractAvg(m1, m2):
    m3 = []
    for i in range(len(m1)):
        m3.append(m2[i]-m1[i]) 
    return m3

#Don't think I used this one either
def matSubtractAvg2(m1, m2):
    m3 = []
    for i in range(len(m1)):
        m3.append(max(m2[i]-m1[i],0)) 
    return m3

#Converts an image vector (or a "column" vector) back into a 2-dimensional list of tuples
def matAddTo2dImgList(m):
    newm = []
    for i in range(int(len(m)/3)):
        newm.append((m[3*i],m[3*i+1],m[3*i+2]))
    chunks = [newm[x:x+376] for x in range(0, len(newm), 376)]
    return chunks

#does the same but also includes the minimum value from the vector for making certain images
def matAddTo2dImgList2(m):
    minim = min(m)
    newm = []
    for i in range(int(len(m)/3)):
        newm.append((m[3*i],m[3*i+1],m[3*i+2]))
    chunks = [newm[x:x+376] for x in range(0, len(newm), 376)]
    return [chunks,minim]

#takes a grayscale image vector (or list of brightness values) and returns a 2-dimensional list of tuples that can be made into an image
def makeimg(d):
    matty = []
    h = len(d)
    w = len(d[0])
    for i in range(h):
        matty.append([])
        for j in range(w):
            matty[i].append((int(d[j][i][0]*255), int(d[j][i][1]*255), int(d[j][i][2]*255), 255))
    return matty

#this was used for testing in prior versions, it does the same as above but makes a sort of combination of two images and also ensures that any negative values in the RGB tuples are replaced with 0 for the process of actually making an image
def makeimg2(d,e):
    matty=[]
    h=len(d)
    w=len(d[0])
    for i in range(h):
        matty.append([])
        for j in range(w):
            matty[i].append((max(int((e[j][i][0]-d[j][i][0])*255),0), max(int((e[j][i][1]-d[j][i][1])*255),0), max(int((e[j][i][2]-d[j][i][2])*255),0), 255))
    return matty

#same as above but this assumes the images aren't grayscale and the rgb tuples have integer values between 0 and 255
def makeimg3(d,e):
    matty=[]
    h=len(d)
    w=len(d[0])
    for i in range(h):
        matty.append([])
        for j in range(w):
            matty[i].append((max(int((e[j][i][0]-d[j][i][0])),0), max(int((e[j][i][1]-d[j][i][1])),0), max(int((e[j][i][2]-d[j][i][2])),0), 255))
    return matty

#this is for a single (color) image
def makeimg4(d):
    matty=[]
    h=len(d)
    w=len(d[0])
    for j in range(w):
        matty.append([])
        for i in range(h):
            matty[j].append((max(int((d[i][j][0])),0), max(int((d[i][j][1])),0), max(int((d[i][j][2])),0), 255))
    return matty    

#this was me trying to make the images brighter so they weren't like 90% black
def makeimg5(d):
    matty=[]
    h=len(d)
    w=len(d[0])
    for i in range(w):
        matty.append([])
        for j in range(h):
            matty[i].append((max(min(int((d[j][i][0])+100),255),0), max(min(int((d[j][i][1])+100),255),0), max(min(int((d[j][i][2])+100),255),0), 255))
    return matty 

def makeimg6(d):
    matty=[]
    h=len(d)
    w=len(d[0])
    for j in range(w):
        matty.append([])
        for i in range(h):
            matty[j].append((int((d[i][j][0])), int((d[i][j][1])), int((d[i][j][2])), 255))
    return matty    

def makeimg7(d,e):
    matty=[]
    h=len(d)
    w=len(d[0])
    for j in range(w):
        matty.append([])
        for i in range(h):
            matty[j].append((int((d[i][j][0])+abs(e)), int((d[i][j][1])+abs(e)), int((d[i][j][2])+abs(e)), 255))
    return [matty,e]

#this imports all the images in the folder so that they can be used for testing
bonnie1 = tomatlist(getdata(Image.open("bonnie1.jpg")))[0]
bonnie2 = tomatlist(getdata(Image.open("bonnie2.jpg")))[0]
boomer1 = tomatlist(getdata(Image.open("boomer1.jpg")))[0]
boomer2 = tomatlist(getdata(Image.open("boomer2.jpg")))[0]
boomer3 = tomatlist(getdata(Image.open("boomer3.jpg")))[0]
boomer4 = tomatlist(getdata(Image.open("boomer4.jpg")))[0]
boomer5 = tomatlist(getdata(Image.open("boomer5.jpg")))[0]
fire1 = tomatlist(getdata(Image.open("fire1.jpg")))[0]
fire2 = tomatlist(getdata(Image.open("fire2.jpg")))[0]
fire3 = tomatlist(getdata(Image.open("fire3.jpg")))[0]
fire4 = tomatlist(getdata(Image.open("fire4.jpg")))[0]
jake1 = tomatlist(getdata(Image.open("jake1.jpg")))[0]
jake2 = tomatlist(getdata(Image.open("jake2.jpg")))[0]
jake3 = tomatlist(getdata(Image.open("jake3.jpg")))[0]
naya1 = tomatlist(getdata(Image.open("naya1.jpg")))[0]
peezer1 = tomatlist(getdata(Image.open("peezer1.jpg")))[0]
peezer2 = tomatlist(getdata(Image.open("peezer2.jpg")))[0]
peezer3 = tomatlist(getdata(Image.open("peezer3.jpg")))[0]
peezer4 = tomatlist(getdata(Image.open("peezer4.jpg")))[0]
slade1 = tomatlist(getdata(Image.open("slade1.jpg")))[0]
slade2 = tomatlist(getdata(Image.open("slade2.jpg")))[0]
slade3 = tomatlist(getdata(Image.open("slade3.jpg")))[0]
slade4 = tomatlist(getdata(Image.open("slade4.jpg")))[0]
solano1 = tomatlist(getdata(Image.open("solano1.jpg")))[0]
tigera1 = tomatlist(getdata(Image.open("tigera1.jpg")))[0]
tigerb1 = tomatlist(getdata(Image.open("tigerb1.jpg")))[0]
tigerb2 = tomatlist(getdata(Image.open("tigerb2.jpg")))[0]
tigerc1 = tomatlist(getdata(Image.open("tigerc1.jpg")))[0]
tigerd1 = tomatlist(getdata(Image.open("tigerd1.jpg")))[0]
tigerd2 = tomatlist(getdata(Image.open("tigerd2.jpg")))[0]
tigere1 = tomatlist(getdata(Image.open("tigere1.jpg")))[0]
sunglassslade1 = tomatlist(getdata(Image.open("sunglassslade1.jpg")))[0]
uglybeebeezoom = tomatlist(getdata(Image.open("uglybeebeezoom.jpg")))[0]

#this is the test set, arranged into a 13 x 338400 matrix
tgroup = np.transpose(np.matrix([bonnie1, boomer1, fire1, jake1, naya1, peezer1, slade1, solano1, tigera1, tigerb1, tigerc1, tigerd1, tigere1]))

#this is the vector that gives us the average of the thirteen faces
avgface = ((1/13)*(np.matrix(bonnie1)+np.matrix(boomer1)+np.matrix(fire1)+np.matrix(jake1)+np.matrix(naya1)+np.matrix(peezer1)+np.matrix(slade1)+np.matrix(solano1)+np.matrix(tigera1)+np.matrix(tigerb1)+np.matrix(tigerc1)+np.matrix(tigerd1)+np.matrix(tigere1))).tolist()

#this is my way of working around an issue with numpy that I couldn't figure out
aface = np.transpose(np.matrix([avgface[0], avgface[0], avgface[0], avgface[0], avgface[0], avgface[0], avgface[0], avgface[0], avgface[0], avgface[0], avgface[0], avgface[0], avgface[0]]))

#the recentered/normalized matrix of the test set image vectors
eXes = (1/np.sqrt(13))*(tgroup - aface)

#The transpose of eXes multiplied by eXes
XtX = np.transpose(eXes)*eXes

#this gives us the eigenvectors of XtX
eevees = np.linalg.eig(XtX)[1]

#getting the teigernfaces by multipling the eXes matrix with the eevees matrix
Yiii = eXes*eevees

#this is just to make it easier to retrieve the data for later use
bb = np.transpose(Yiii)

#converting the teigernfaces to lists for ease of use
y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13 = bb.tolist()

#making a 2dlist of the teigernfaces to make my life easier
yiiis = [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13]

#get the coefficients of an image to make a weighted sum
def get_coeffs(face):
    fce1 = np.matrix(face)-np.matrix(avgface)
    d1 = (np.dot(fce1, np.transpose(np.matrix(y1)))/np.dot(np.matrix(y1),np.transpose(np.matrix(y1)))).tolist()[0][0]
    d2 = (np.dot(fce1, np.transpose(np.matrix(y2)))/np.dot(np.matrix(y2),np.transpose(np.matrix(y2)))).tolist()[0][0]
    d3 = (np.dot(fce1, np.transpose(np.matrix(y3)))/np.dot(np.matrix(y3),np.transpose(np.matrix(y3)))).tolist()[0][0]
    d4 = (np.dot(fce1, np.transpose(np.matrix(y4)))/np.dot(np.matrix(y4),np.transpose(np.matrix(y4)))).tolist()[0][0]
    d5 = (np.dot(fce1, np.transpose(np.matrix(y5)))/np.dot(np.matrix(y5),np.transpose(np.matrix(y5)))).tolist()[0][0]
    d6 = (np.dot(fce1, np.transpose(np.matrix(y6)))/np.dot(np.matrix(y6),np.transpose(np.matrix(y6)))).tolist()[0][0]
    d7 = (np.dot(fce1, np.transpose(np.matrix(y7)))/np.dot(np.matrix(y7),np.transpose(np.matrix(y7)))).tolist()[0][0]
    d8 = (np.dot(fce1, np.transpose(np.matrix(y8)))/np.dot(np.matrix(y8),np.transpose(np.matrix(y8)))).tolist()[0][0]
    d9 = (np.dot(fce1, np.transpose(np.matrix(y9)))/np.dot(np.matrix(y9),np.transpose(np.matrix(y9)))).tolist()[0][0]
    d10 = (np.dot(fce1, np.transpose(np.matrix(y10)))/np.dot(np.matrix(y10),np.transpose(np.matrix(y10)))).tolist()[0][0]
    d11 = (np.dot(fce1, np.transpose(np.matrix(y11)))/np.dot(np.matrix(y11),np.transpose(np.matrix(y11)))).tolist()[0][0]
    d12 = (np.dot(fce1, np.transpose(np.matrix(y12)))/np.dot(np.matrix(y12),np.transpose(np.matrix(y12)))).tolist()[0][0]
    d13 = (np.dot(fce1, np.transpose(np.matrix(y13)))/np.dot(np.matrix(y13),np.transpose(np.matrix(y13)))).tolist()[0][0]  
    return [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13]

#reconstruct an image with a given amount of teigernfaces
def reconst_face_with_coeffs(face,amnt):
    dd = get_coeffs(face)
    newface = avgface
    for i in range(amnt):
        newface = newface + np.multiply(dd[i],yiiis[i])
    return newface

#list of all the coefficients of the test set for the sake of comparison
setcoeffs = [get_coeffs(bonnie1),get_coeffs(boomer1),get_coeffs(fire1),get_coeffs(jake1),get_coeffs(naya1),get_coeffs(peezer1),get_coeffs(slade1),get_coeffs(solano1),get_coeffs(tigera1),get_coeffs(tigerb1),get_coeffs(tigerc1),get_coeffs(tigerd1),get_coeffs(tigere1)]

#find the sum of the differences between the coefficients of two images
def sum_of_diffs(face1,face2):
    difff = 0
    for i in range(len(face1)):
        if i != 1:
            difff += abs(face1[i]-face2[i])
    return difff

#compare an image to the images in the test set
def compare_coeffs(face):
    dd = get_coeffs(face)
    graph = []
    for i in setcoeffs:
        graph.append(sum_of_diffs(dd,i))
    return graph


#returns a sorted list based on which tiger is closest to the one in the image given
def get_closest_match(face):
    names = ["Bonnie", "Boomer", "Fire", "Jake", "Naya", "Peezer", "Slade", "Solano", "TigerA", "TigerB", "TigerC", "TigerD", "TigerE"]
    dd=compare_coeffs(face)
    pairup = list(zip(names,dd))
    return sorted(pairup, key = lambda x: x[1])

def make_a_graph(face):
    waa = compare_coeffs(face)
    fart = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    names = ["Bonnie", "Boomer", "Fire", "Jake", "Naya", "Peezer", "Slade", "Solano", "TigerA", "TigerB", "TigerC", "TigerD", "TigerE"]
    fig, ax = plt.subplots()
    ax.scatter(fart,waa)
    for i, txt in enumerate(names):
        ax.annotate(txt, (fart[i],waa[i]))
    return True


#saves an image after a reconstruction
def saveimg(x, imgnape):
    tx = makeimg6(matAddTo2dImgList(x[0]))
    arra = np.array(tx, dtype = np.uint8)
    simg = Image.fromarray(arra)
    simg.save(imgnape)
    return True

def saveimg2(x, imgnape):
    tx,minn = makeimg7(matAddTo2dImgList2(x)[0],matAddTo2dImgList2(x)[1])
    arra = np.array(tx, dtype = np.uint8)
    simg = Image.fromarray(arra)
    simg.save(str(minn) + imgnape)
    return True
