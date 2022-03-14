import numpy as np

class TGT_IMG:
    'Gen target zeros image by scalor'
    #def __init__(self,img,scalor,img_w,img_h,tgt_img,tgt_w,tgt_h):
    def __init__(self,img,tgt_w,tgt_h):
        self.img      = img
        self.img_w    = 0
        self.img_h    = 0
        self.tgt_img  = 0
        self.tgt_w    = tgt_w
        self.tgt_h    = tgt_h

    def execute(self):    
        self.img_w    = self.img.shape[0]
        self.img_h    = self.img.shape[1]
        if self.img.ndim == 3:
            self.tgt_img  = np.zeros((self.tgt_w,self.tgt_h,self.img.shape[2]),dtype=np.uint8)
        else:
            self.tgt_img  = np.zeros((self.tgt_w,self.tgt_h),dtype=np.uint8)
        return self
