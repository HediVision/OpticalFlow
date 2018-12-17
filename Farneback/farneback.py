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

def get2DGuassian(size,sigma):
        Kernelx = cv.getGaussianKernel(size, sigma, ktype=cv.CV_64F)#x direction
        Kernely = cv.getGaussianKernel(size, sigma, ktype=cv.CV_64F)#y direction
        g= Kernelx*np.transpose(Kernely)#2D
        return g.flatten()

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
        A1_poly = np.zeros((w,h,2,2),dtype=np.float64)
        A2_poly = np.zeros((w,h,2,2),dtype=np.float64)
        b1_poly = np.zeros((w,h,2),dtype=np.float64)
        b2_poly = np.zeros((w,h,2),dtype=np.float64)
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
                        b2 = [r1[1],
                        r1[2]]
                        A1_poly[y,x,:,:] =A1
                        A2_poly[y,x,:,:] =A2
                        b1_poly[y,x,:]   =b1
                        b2_poly[y,x,:]   =b2
        return A1_poly,A2_poly,b1_poly,b2_poly

f1=cv.imread('frame-1.png',0)
f2=cv.imread('frame-2.png',0)
w,h=f1.shape
f1=np.float64(f1)
f2=np.float64(f2)
ksize=11
pad=int(np.floor(ksize/2))
sigma=1.5
f1 = cv.copyMakeBorder(f1,pad,pad,pad,pad,cv.BORDER_CONSTANT,value=0)
f2 = cv.copyMakeBorder(f2,pad,pad,pad,pad,cv.BORDER_CONSTANT,value=0)

c = getCertaintyMatrix(f1.shape[0],f1.shape[1])
c = cv.copyMakeBorder(c,pad,pad,pad,pad,cv.BORDER_CONSTANT,value=0)
a = get2DGuassian(ksize,sigma)
B = getBases(ksize)
A1,A2,b1,b2 = getPoly(B,a,c,f1,f2,ksize)

wsize = 39;   
wsigma = 6;    
w1 = cv.getGaussianKernel(wsize, wsigma, ktype=cv.CV_64F)
A_poly = np.zeros((2,2),np.float64)       #Quadratic A matrix
dB_poly = np.zeros((2,1),np.float64)      #Quadratic b
AtA_poly = np.zeros((h, w,2,2),np.float64)
AtB_poly = np.zeros((h, w,2),np.float64)
AtA_conv = np.zeros((h, w,2,2),np.float64)
AtB_conv = np.zeros((h, w,2),np.float64)
dis = np.zeros((f1.shape[0],f1.shape[1],2),np.float64)


             
              

	
	


