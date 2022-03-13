import numpy as np
import cv2
import matplotlib as plt


from model.img_tgt import TGT_IMG

bmp_path = './jpg/LenaRGB.bmp'
tif_path = './jpg/LenaRGB.tif'

img = cv2.imread(bmp_path,0)

cv2.imshow("my_win",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
