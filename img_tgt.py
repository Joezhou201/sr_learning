import numpy as np

class TGT_IMG:
    'Gen taget image size, fill in with 0 if sclar > 1'
    def __init__(self,img,scalor,img_w,img_h,tgt_w,tgt_h,tgt_img):
        self.img         = img
        self.scalor      = scalor
        self.img_w       = img_w    
        self.img_h       = img_h    
        self.tgt_w       = tgt_w    
        self.tgt_h       = tgt_h    
        self.tgt_img     = tgt_img  








    
    def excute(self):
        img       = self.img
        scalor    = self.scalor
        img_w     = self.img.shape[0]
        img_h     = self.img.shape[1]
        tgt_w     = np.int(np.floor(img_w*scalor))
        tgt_h     = np.int(np.floor(img_h*scalor))
        tgt_img   = np.zeros((tgt_w,tgt_h),dtype=int16)
