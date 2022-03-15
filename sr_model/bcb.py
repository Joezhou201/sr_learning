import numpy as np

class BCB:
    'apply bicubic  interplolation algorithm to img'

    def __init__(self,img,img_w,img_h,tgt_img,tgt_w,tgt_h):
        self.img      = img
        self.img_w    = img_w   
        self.img_h    = img_h   
        self.tgt_img  = tgt_img 
        self.tgt_w    = tgt_w   
        self.tgt_h    = tgt_h   
    
    def core_func(x,a):
        if abs(x) <= 1:
            w = (a+2)*abs(x)**3-(a+3)*abs(x)**2+1
        elif abs(x) <= 2:
            w = a*abs(x)**3-5*a*abs(x)**2+8*a*abs(x)-4*a
        else:
            w = 0
        return w    

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

                    bln_x1 = self.img[llx  ,lly]*llu + self.img[llx+1,lly  ]*(1-llu)
                    bln_x2 = self.img[llx+1,lly]*llu + self.img[llx+1,lly+1]*(1-llu)
                    bln_y  = bln_x1*llv              + bln_x2*(1-llv)

                    self.tgt_img[i,j]  = np.clip(round(bln_y),0,255)


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

                        bln_x1 = self.img[llx  ,lly,k]*llu + self.img[llx+1,lly  ,k]*(1-llu)
                        bln_x2 = self.img[llx+1,lly,k]*llu + self.img[llx+1,lly+1,k]*(1-llu)
                        bln_y  = bln_x1*llv              + bln_x2*(1-llv)

                        self.tgt_img[i,j,k]  = np.clip(round(bln_y),0,255)
        return self



