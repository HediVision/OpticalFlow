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
def isOutOfImage(x,y,img):
    rows,cols=img.shape
    if(x >= 0 and x < cols-1 and y >= 0 and y < rows-1):
        return True
    else:
        return False

def get_LocalAvg(im):
    #kernel=np.array([[0, 1/4, 0],
                     #[1/4, 0, 1/4 ],
                     #[0, 1/4, 0]], dtype=np.float64)
    kernel=np.array([[1/12, 1/6, 1/12],
                     [1/6, 0, 1/6 ],
                     [1/12, 1/6, 1/12]], dtype=np.float64)
    avg=cv.filter2D(im,-1,kernel)
    return avg

def get_Ix(I_1,I_2):
    kernel=np.ones((2,2), dtype = np.float64)
    kernel[:,0]=-1.0
    g_x1=cv.filter2D(I_1,-1,kernel)
    g_x2=cv.filter2D(I_2,-1,kernel)
    Ix=g_x1+g_x2
    return Ix

def get_Iy(I_1,I_2):
    kernel=np.ones((2,2),dtype=np.float64)
    kernel[0,:]=-1.0
    g_y1=cv.filter2D(I_1,-1,kernel)
    g_y2=cv.filter2D(I_2,-1,kernel)
    Iy=g_y1+g_y2
    return Iy

def get_It(I_1,I_2):
    kernel=np.ones((2,2),dtype=np.float64)*-1
    g_t1=cv.filter2D(I_1,-1,kernel)
    g_t2=cv.filter2D(I_2,-1,(-kernel))
    It = g_t1+g_t2
    return It

def runHornShunck(I_1,I_2):
    lmda = 0.05
    u =  np.zeros(I_1.shape, dtype=I_1.dtype)
    v =  np.zeros(I_1.shape, dtype=I_1.dtype)
    Ix=get_Ix(I_1,I_2)
    Iy=get_Iy(I_1,I_2)
    It=get_It(I_1,I_2)
    itr=0
    p_err=0
    while True:
        u_avg=get_LocalAvg(u)
        v_avg=get_LocalAvg(v)
        P=Ix * u_avg + Iy * v_avg + It
        D=Ix*Ix + Iy*Iy+ lmda
        der=P/D
        u = u_avg - Ix*der
        v = v_avg - Iy*der
        err = (Ix*u + Iy*v + It).mean()
        print('itr %i : %f %f'% (itr,err,p_err))
        if(itr != 0 and np.abs(p_err) <= np.abs(err)):
            break
        itr+=1
        p_err=err
 
    rows,cols=I_1.shape
    f = plt.figure()         
    plt.imshow(I_2,cmap='gray')
    plt.xticks([]), plt.yticks([])
    for y in range(0, rows-1,25):
        for x in range(0, cols-1,25):
            fx=u[y,x]*25
            fy=v[y,x]*25
            plt.arrow(x,y,fx,fy,head_width=4, head_length=4,fc='r', ec='r')
    plt.show()
   



I_1=cv.imread('images/frame_1.tif',0)
I_2=cv.imread('images/frame_2.tif',0)
I_1=np.float64(I_1)*(1.0/255)
I_2=np.float64(I_2)*(1.0/255)
runHornShunck(I_1,I_2)
