import numpy as np

class NNB:
    'apply nearest neighbor algorithm to img'

    def __init__(self,img,img_w,img_h,tgt_img,tgt_w,tgt_h):
        self.img      = img
        self.img_w    = img_w   
        self.img_h    = img_h   
        self.tgt_img  = tgt_img 
        self.tgt_w    = tgt_w   
        self.tgt_h    = tgt_h   
    
    def execute(self):
        scalor_x  = (self.tgt_w-1)/(self.img_w-1)
        scalor_y  = (self.tgt_h-1)/(self.img_h-1)

        if self.img.ndim == 2:
            self.img  = np.pad(self.img,((0,1),(0,1)),'edge')
            for i in range(self.tgt_w):
                for j in range(self.tgt_h):
                    ix    = i/scalor_x
                    llx   = np.int(i//scalor_x)
                    llu   = ix - llx

                    jy    = j/scalor_y
                    lly   = np.int(j//scalor_y)
                    llv   = jy - lly

                    ll    = llu     + llv
                    lh    = llu     + (1-llv)
                    hl    = (1-llu) + llv
                    hh    = (1-llu) + (1-llv)
                    min_d = np.argmin([ll,lh,hl,hh])
                    if min_d == 0:
                        self.tgt_img[i,j] = self.img[llx  ,lly  ]
                    elif min_d == 1:
                        self.tgt_img[i,j] = self.img[llx  ,lly+1]
                    elif min_d == 2:
                        self.tgt_img[i,j] = self.img[llx+1,lly  ]
                    else:           
                        self.tgt_img[i,j] = self.img[llx+1,lly+1]
        elif self.img.ndim == 3:
            for k in range(self.img.shape[2]):
                for i in range(self.tgt_w):
                    for j in range(self.tgt_h):
                        ix    = i/scalor_x
                        llx   = np.int(i//scalor_x)
                        llu   = ix - llx

                        jy    = j/scalor_y
                        lly   = np.int(j//scalor_y)
                        llv   = jy - lly

                        ll    = llu     + llv
                        lh    = llu     + (1-llv)
                        hl    = (1-llu) + llv
                        hh    = (1-llu) + (1-llv)
                        min_d = np.argmin([ll,lh,hl,hh])
                        if min_d == 0:
                            self.tgt_img[i,j,k] = self.img[llx  ,lly  ,k]
                        elif min_d == 1:
                            self.tgt_img[i,j,k] = self.img[llx  ,lly+2,k]
                        elif min_d == 2:
                            self.tgt_img[i,j,k] = self.img[llx+1,lly  ,k]
                        else:           
                            self.tgt_img[i,j,k] = self.img[llx+1,lly+1,k]
        return self



