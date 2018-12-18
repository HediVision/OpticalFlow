import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 

def getBases(size,type=np.float64):
        B = np.ones((size*size,6),dtype=type)
        row=np.ones((1,size),dtype=type)
        cols=np.ones((size,1),dtype=type)
        for i in range (0,size):
            row[0,i] = i-np.floor(size/2)
            cols[i,0] = i-np.floor(size/2)
        B[...,1] = cv.repeat(row,size,1).flatten()#x
        B[...,2] = cv.repeat(cols,1,size).flatten()#y
        B[...,3]=B[...,1]*B[...,1]#x^2
        B[...,4]=B[...,2]*B[...,2]#y^2
        B[...,5]=B[...,1]*B[...,2]#xy
        return B

def get2DGuassian(size,sigma,flatten=True):
        Kernelx = cv.getGaussianKernel(size, sigma, ktype=cv.CV_64F)#x direction
        Kernely = cv.getGaussianKernel(size, sigma, ktype=cv.CV_64F)#y direction
        g= Kernelx*np.transpose(Kernely)#2D
        if flatten:
                return g.flatten()
        else:
                return g

def getCertaintyMatrix(rows,cols,type=np.float64):
        c=np.ones((rows-12,cols-12), dtype=type)
        c=cv.copyMakeBorder(c,1,1,1,1,cv.BORDER_CONSTANT,value=0.80)
        c=cv.copyMakeBorder(c,1,1,1,1,cv.BORDER_CONSTANT,value=0.70)
        c=cv.copyMakeBorder(c,1,1,1,1,cv.BORDER_CONSTANT,value=0.60)
        c=cv.copyMakeBorder(c,1,1,1,1,cv.BORDER_CONSTANT,value=0.40)
        c=cv.copyMakeBorder(c,1,1,1,1,cv.BORDER_CONSTANT,value=0.20)
        c=cv.copyMakeBorder(c,1,1,1,1,cv.BORDER_CONSTANT,value=0.10)
        return c

def getPoly(B,a,c,f1,f2,size):
        Wa=np.diag(a)
        BTWa=np.matmul(B.T,Wa)
        A1_poly = np.zeros((h,w,2,2),dtype=np.float64)
        A2_poly = np.zeros((h,w,2,2),dtype=np.float64)
        b1_poly = np.zeros((h,w,2),dtype=np.float64)
        b2_poly = np.zeros((h,w,2),dtype=np.float64)
        windowSize=(size, size)
        for y in range(0, f1.shape[0], 1):
                for x in range(0, f1.shape[1], 1):
                        p_f1 = f1[y:y + windowSize[1], x:x + windowSize[0]].flatten()
                        p_f2 = f2[y:y + windowSize[1], x:x + windowSize[0]].flatten()
                        Wc  = np.diag(c[y:y + windowSize[1], x:x + windowSize[0]].flatten())
                        if(p_f1.shape[0]!=size*size):
                                continue
                        BTWaWc  = np.matmul(BTWa,Wc)
                        BTWaWcB = np.matmul(BTWaWc,B)
                        #Solve systems of linear equations Ax = B for x
                        _,r1 = cv.solve(BTWaWcB, np.matmul(BTWaWc,p_f1))
                        _,r2 = cv.solve(BTWaWcB, np.matmul(BTWaWc,p_f2))
                        A1 = [ [r1[3], r1[5]/2],
                        [r1[5]/2, r1[4]] ]
                        A2 = [ [r2[3], r2[5]/2],
                        [r2[5]/2, r2[4]] ]
                        b1 = [r1[1],
                        r1[2]]
                        b2 = [r2[1],
                        r2[2]]
                        A1_poly[y,x,:,:] =A1
                        A2_poly[y,x,:,:] =A2
                        b1_poly[y,x,:]   =b1
                        b2_poly[y,x,:]   =b2
        return A1_poly,A2_poly,b1_poly,b2_poly

f1=cv.imread('frame-1.png',0)
f2=cv.imread('frame-2.png',0)
f1=np.float64(f1)
f2=np.float64(f2)
h,w=f1.shape
ksize=11
sigma=1.5
pad=int(np.floor(ksize/2))
#Signal
f1 = cv.copyMakeBorder(f1,pad,pad,pad,pad,cv.BORDER_CONSTANT,value=0)
f2 = cv.copyMakeBorder(f2,pad,pad,pad,pad,cv.BORDER_CONSTANT,value=0)

#Certainty Matrix
c = getCertaintyMatrix(f1.shape[0],f1.shape[1])
c = cv.copyMakeBorder(c,pad,pad,pad,pad,cv.BORDER_CONSTANT,value=0)
#Applicability matrix 

a = get2DGuassian(ksize,sigma)
#Bases
B = getBases(ksize)
#get poly
A1,A2,b1,b2 = getPoly(B,a,c,f1,f2,ksize)

#weight matrix 
wsize = 21;   
wsigma = 5; 
weight=get2DGuassian(wsize,wsigma,flatten=False)   

A = np.zeros((2,2),np.float64)  
b_hat = np.zeros((2,1),np.float64)
AtA = np.zeros((h, w,2,2),np.float64)
AtB = np.zeros((h, w,2,1),np.float64)
AtA_conv = np.zeros((h, w,2,2),np.float64)
AtB_conv = np.zeros((h, w,2),np.float64)
dis = np.zeros((f1.shape[0],f1.shape[1],2,1),np.float64)


for y1 in range(0,h):
        for x1 in range(0,w):
                d=np.reshape(dis[y1,x1,:], (2,1), order="F")
                y2 = int(np.round(y1+d[1]))
                x2 = int(np.round(x1+d[0]))
                if y2 > h:
                        y2 = h-1
                elif y2 < 0:
                        y2= 0
                if x2 > w:
                        x2 = w-1
                elif x2 < 0:
                        x2= 0
                p_A1 = np.reshape(A1[y1,x1,:,:], (2,2)) 
                p_A2 = np.reshape(A2[y2,x2,:,:], (2,2), order="F") 
                p_b1 = np.reshape(b1[y1,x1,:], (2,1))
                p_b2 = np.reshape(b2[y2,x2,:], (2,1))
                A = (p_A2 + p_A1)/2
                b_hat= (-(p_b2- p_b1)/2) + np.matmul(A,d)
                AtA[y1, x1,:,:] = np.matmul((A.T),(A))
                x=np.matmul((A.T),b_hat)
                AtB[y1, x1,:,:] = np.matmul((A.T),b_hat)
for i in range (0,2):
        for j in range (0,2):
                temp = np.reshape(AtA[:,:,i,j], (h, w), order="F")
                AtA_conv[:,:,i,j]= cv.filter2D(temp,-1,weight,cv.BORDER_CONSTANT)
        temp2 = np.reshape(AtB[:,:,i], (h, w), order="F")
        AtB_conv[:,:,i] = cv.filter2D(temp2,-1,weight,cv.BORDER_CONSTANT)
for y in range(0,h):
        for x in range(0,w):
                AtA = np.reshape(AtA_conv[y, x,:,:], (2, 2))
                AtB = np.reshape(AtB_conv[y, x,:], (2, 1))
                _,dis[y, x,:] = cv.solve(AtA,AtB)

f = plt.figure()         
plt.imshow(f1,cmap='gray')
plt.xticks([]), plt.yticks([])
for y in range(0, h-1,10):
        for x in range(0, w-1,10):
            fx=float(dis[y,x,0]*25)
            fy=float(dis[y,x,1]*25)
            plt.arrow(x,y,fx,fy,head_width=4, head_length=4,fc='r', ec='r')
plt.show()
             
                

 
 





             
              

	
	


