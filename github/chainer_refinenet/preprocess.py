
import numpy as np
from PIL import Image
import random

def load_data(path, crop=False, mode="label", xp=np, hflip=False, rscale=False, rcrop=False, xs=0, ys=0, rs=256):

  '''
  1. Pickrandom L in range[256,480]
  2. Resize training image, shortside=L
  3. Sample random 224x224 patch
  '''

  try:
      img = Image.open(path)
  except FileNotFoundError:
      # print("file not found : {}".format(path))
      img = Image.open("/home/ppdev/data/pictures/Pennsylvania/39.7212756378488,-80.51015989198811.png")
      return None
#   if img.mode == 'L':
#       img = img.convert('RGB')

  if crop:
    w, h = img.size
    if rcrop:
      size = rs
    else:
      size = 224

    if w < h:
      img = img.resize((size, size*h//w))
      w, h = img.size
      if not rcrop:
        xs = 0
        ys = (h-224)//2

    else:
      img = img.resize((size*w//h, size))
      w, h = img.size
      if not rcrop:
        xs = (w-224)//2
        ys = 0

  if mode=="label":
    y = xp.asarray(img, dtype=xp.int32)
    if np.sum(y==1)<10:
        return None
    y = y[ys:ys+224,xs:xs+224]

    if hflip:
      y = y[:,::-1]

    mask = y == 255
    y[mask] = -1
#     print(y)

    return y

  elif mode=="data":
    mean = xp.array([103.939, 116.779, 123.68])
#     print(img.size, img.mode)
    if img.mode == 'L' or img.mode == 'P':
      rgbimg = Image.new("RGB", img.size)
      rgbimg.paste(img)
      img = rgbimg


#     print(img.size, img.mode)
#     img -= mean
    x = xp.asarray(img, dtype=xp.float32)
#     print(x.shape)
    x -= mean
    x = x.transpose(2, 0, 1)
#     print(x.shape)
    x = x[:,ys:ys+224,xs:xs+224]

#     print(x.shape)


    if hflip:
      x = x[:,:,::-1]

    return x
