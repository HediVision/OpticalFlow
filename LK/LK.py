import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt 
def implot(imgs):
    n=1
    for i in imgs:
        plt.subplot(2,2,n),plt.imshow(i, cmap='gray')
        plt.xticks([]), plt.yticks([])
        n+=1
    plt.show()

def get_Ix(I_1,I_2):
    kernel = np.ones((2,2), dtype = np.float64)
    kernel[:,0] = -1.0
    g_x1 = cv.filter2D(I_1,-1,kernel)
    g_x2 = cv.filter2D(I_2,-1,kernel)
    Ix   = g_x1+g_x2
    return Ix

def get_Iy(I_1,I_2):
    kernel = np.ones((2,2),dtype=np.float64)
    kernel[0,:] = -1.0
    g_y1 = cv.filter2D(I_1,-1,kernel)
    g_y2 = cv.filter2D(I_2,-1,kernel)
    Iy   = g_y1+g_y2
    return Iy

def get_It(I_1,I_2):
    kernel=np.ones((2,2),dtype=np.float64)*-1
    g_t1 = cv.filter2D(I_1,-1,kernel)
    g_t2 = cv.filter2D(I_2,-1,(-kernel))
    It = g_t1+g_t2
    return It

def get_localsum(im):
    kernel = np.ones((3,3),np.float64)
    sum = cv.filter2D(im,-1,kernel)
    sum = cv.GaussianBlur(sum,(3,3),0)
    return sum   

def runLK(I_1,I_2):
    eps= 2.220446049250313e-16
    u  = np.zeros(I_1.shape, dtype=I_1.dtype)
    v  = np.zeros(I_1.shape, dtype=I_1.dtype)
    Ix = get_Ix(I_1,I_2)
    Iy = get_Iy(I_1,I_2)
    It = get_It(I_1,I_2)
    
    Ix2  = Ix*Ix
    Iy2  = Iy*Iy
    IxIy = Iy*Ix
    IxIt = Ix*It
    IyIt = Iy*It
    
    sum_Ix2  = get_localsum(Ix2)
    sum_Iy2  = get_localsum(Iy2)
    sum_IxIy = get_localsum(IxIy)
    sum_IxIt = get_localsum(IxIt)
    sum_IyIt = get_localsum(IyIt)
    tmp = (sum_Ix2*sum_Iy2)-(sum_IxIy*sum_IxIy)+eps
    
    u = (sum_IxIy*sum_IyIt)-(sum_Iy2*sum_IxIt)
    v = (sum_IxIt*sum_IxIy)-(sum_Ix2*sum_IyIt)
    u = u/tmp
    v = v/tmp
    return [u,v]

I_1=cv.imread('../images/frame_1.tif',0)
I_2=cv.imread('../images/frame_2.tif',0)
I_1=np.float64(I_1)*(1.0/255)
I_2=np.float64(I_2)*(1.0/255)
[u,v]=runLK(I_1,I_2)
rows,cols=I_1.shape
f = plt.figure()         
plt.imshow(I_2,cmap='gray')
plt.xticks([]), plt.yticks([])
for y in range(0, rows-1,25):
    for x in range(0, cols-1,25):
        fx=u[y,x]*25
        fy=v[y,x]*25
        plt.arrow(x,y,fx,fy,head_width=8, head_length=8,fc='r', ec='r')
plt.show()
