import numpy as np
import cv2
from matplotlib import pyplot as plt


from sr_model.tgt_img import TGT_IMG
from sr_model.nnb     import NNB
from sr_model.bln     import BLN


bmp_path = './jpg/LenaRGB.bmp'
tif_path = './jpg/LenaRGB.tif'

########cv2 read color format is BGR, need to change to RGB
img    = cv2.imread(bmp_path,0)
scalor =  0.6
tgt_w  = np.int(np.floor(img.shape[0]*scalor))
tgt_h  = np.int(np.floor(img.shape[1]*scalor))

img_lx  = 22
img_hx  = 30
img_ly  = 22
img_hy  = 30
if img.ndim == 3:
    img_cut   = img[img_lx:img_hx,img_ly:img_hy,[2,1,0]]
    img       = img[:,:,[2,1,0]]
else:    
    img_cut   = img[img_lx:img_hx,img_ly:img_hy]
#cv2.imshow("my_win",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


######gen zero array for target image############
tgt_img_ex = TGT_IMG(img,tgt_w,tgt_h)
tgt_img_zr = tgt_img_ex.execute()

img_w    = tgt_img_zr.img_w   
img_h    = tgt_img_zr.img_h   
tgt_img  = tgt_img_zr.tgt_img 
tgt_w    = tgt_img_zr.tgt_w   
tgt_h    = tgt_img_zr.tgt_h   

print(img_cut.shape,img.shape,tgt_img.shape,img_w,img_h,tgt_w,tgt_h)
nnb      = NNB(img,img_w,img_h,tgt_img,tgt_w,tgt_h)
nnb_ex   = nnb.execute()
img_nnb  = nnb_ex.tgt_img
print(img_nnb.shape)

bln      = BLN(img,img_w,img_h,tgt_img,tgt_w,tgt_h)
bln_ex   = bln.execute()
img_bln  = bln_ex.tgt_img
print(img_bln.shape)

dpi  = 80
figsize_org  = img_w/dpi, img_h/dpi
figsize_tgt  = tgt_w/dpi, tgt_h/dpi
plt.figure('orign img',figsize=figsize_org)
if img.ndim == 3:
    plt.imshow(img)
else:
    plt.imshow(img,cmap='gray')

plt.figure('img nnb',figsize=figsize_tgt)
if img.ndim == 3:
    plt.imshow(img_nnb)
else:
    plt.imshow(img_nnb,cmap='gray')

plt.figure('img bln',figsize=figsize_tgt)
if img.ndim == 3:
    plt.imshow(img_bln)
else:
    plt.imshow(img_bln,cmap='gray')

#plt_img = plt.imshow(img,cmap='gray')
plt.show()

