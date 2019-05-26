import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy import io
import cv2
from skimage import io

ReadPath1 = 'D:/cv2_test/no-mask500'
ReadPath2 = 'D:/cv2_test/mask500'
SavePath = 'D:/cv2_test/change'

def rgbtogrey(img1,img2):
    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    return img1,img2

def elastic_transform(image,image_mask,alpha,sigma,random_state=None):

    # assert len(image.shape)==2

    if random_state is None:
        random_state=np.random.RandomState(None)
    shape=image.shape
    dx = gaussian_filter((np.random.rand(*shape)*2-1),sigma,mode='constant',cval=0)*alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode='constant', cval=0) * alpha
    x,y=np.meshgrid(np.arange(shape[0]),np.arange(shape[1]),indexing='ij')

    indices=np.reshape(x+dx,(-1,1)),np.reshape(y+dy,(-1,1))

    return map_coordinates(image,indices,order=1).reshape(shape),map_coordinates(image_mask,indices,order=1).reshape(shape)

str1=ReadPath1+'/*.jpg'
str2=ReadPath2+'/*_mask.jpg'
coll1 = io.ImageCollection(str1)
coll2 = io.ImageCollection(str2)

for i in range(len(coll1)):
    im,im_mask=rgbtogrey(coll1[i],coll2[i])
    im_t,im_mask_t=elastic_transform(im,im_mask, im.shape[1] * 2, im.shape[1] * 0.08, im.shape[1] * 0.08)
    io.imsave(SavePath+'/'+np.str(i)+'.jpg',im_t)  #循环保存图片
    io.imsave(SavePath+'/'+np.str(i)+'_mask.jpg',im_mask_t)

