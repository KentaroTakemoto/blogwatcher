import chainer
from chainer import cuda, optimizers, serializers, Variable
import numpy as np
from PIL import Image
import os
import argparse

from refinenet import RefineResNet
from color_map import make_color_map

def predict(image, weight, class_num, gpu=-1):
  model = RefineResNet(class_num)
  serializers.load_npz(weight, model)

  if gpu >= 0:
    chainer.cuda.get_device(gpu).use()
    model.to_gpu()
  xp = np if args.gpu < 0 else cuda.cupy

  img = image.resize((224,224))
  rgbimg = Image.new("RGB", img.size)
  rgbimg.paste(img)
  img = rgbimg

  mean = xp.array([103.939, 116.779, 123.68])
#   img -= mean
  x = xp.asarray(img, dtype=xp.float32)
  x -= mean
#   x = xp.expand_dims(x, axis=0)
  x = x.transpose(2, 0, 1)

  x = xp.expand_dims(x, axis=0)

  with chainer.using_config('train', False):
    pred = model(x).data

    return pred

def predict_multi(names, weight, class_num, gpu=-1):
  model = RefineResNet(class_num)
  serializers.load_npz(weight, model)

  if gpu >= 0:
    chainer.cuda.get_device(gpu).use()
    model.to_gpu()
  xp = np if args.gpu < 0 else cuda.cupy

  for i,name in enumerate(names):
      img = Image.open("/home/ppdev/codes/blogwatcher/github/chainer_refinenet/test_images/{}.png".format(name))
      img = img.resize((224,224))
      rgbimg = Image.new("RGB", img.size)
      rgbimg.paste(img)
      img = rgbimg

      mean = xp.array([103.939, 116.779, 123.68])
    #   img -= mean
      x = xp.asarray(img, dtype=xp.float32)
      x -= mean
    #   x = xp.expand_dims(x, axis=0)
      x = x.transpose(2, 0, 1)
      if i==0:
          whole = xp.expand_dims(x, axis=0)
      else:
          whole = np.vstack([whole,xp.expand_dims(x, axis=0)])

  with chainer.using_config('train', False):
    pred = model(whole).data

    return pred


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='RefineNet on Chainer (predict)')
  parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--image_path', '-i', default=None, type=str)
  parser.add_argument('--class_num', '-n', default=21, type=int)
  parser.add_argument('--weight', '-w', default="~/data/weights/test2/chainer_refinenet_tmp.weight", type=str)
  args = parser.parse_args()

  if args.image_path[-3:]=="txt":
      with open(args.image_path,"r") as f:
        ls = f.readlines()
      names = [l.rstrip('\n') for l in ls]
      preds = predict_multi(names, args.weight, args.class_num, args.gpu)
      for i,name in enumerate(names):
        x = preds[i].copy()
        pred = preds[i].argmax(axis=0)

        row, col = pred.shape

        xp = np if args.gpu < 0 else cuda.cupy
        dst = xp.ones((row, col, 3))

        color_map = make_color_map()
        for i in range(args.class_num):
          dst[pred == i] = color_map[i]

        if args.gpu >= 0:
          dst = cuda.to_cpu(dst)
        img = Image.fromarray(np.uint8(dst))

        b,g,r = img.split()
        img = Image.merge("RGB", (r, g, b))

        trans = Image.new('RGBA', img.size, (0, 0, 0, 0))
        w, h = img.size
        for x in range(w):
          for y in range(h):
            pixel = img.getpixel((x, y))
            if (pixel[0] == 0   and pixel[1] == 0   and pixel[2] == 0) or \
               (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
              continue
            trans.putpixel((x, y), pixel)

        trans.save("/home/ppdev/codes/blogwatcher/github/chainer_refinenet/test_images/pred_{}.png".format(name))


  else:
      img = Image.open(args.image_path)
      pred = predict(img, args.weight, args.class_num, args.gpu)
      print(pred.shape)
      x = pred[0].copy()
      pred = pred[0].argmax(axis=0)

      row, col = pred.shape

      xp = np if args.gpu < 0 else cuda.cupy
      dst = xp.ones((row, col, 3))

      color_map = make_color_map()
      for i in range(args.class_num):
        dst[pred == i] = color_map[i]

      if args.gpu >= 0:
        dst = cuda.to_cpu(dst)
      img = Image.fromarray(np.uint8(dst))

      b,g,r = img.split()
      img = Image.merge("RGB", (r, g, b))

      trans = Image.new('RGBA', img.size, (0, 0, 0, 0))
      w, h = img.size
      for x in range(w):
        for y in range(h):
          pixel = img.getpixel((x, y))
          if (pixel[0] == 0   and pixel[1] == 0   and pixel[2] == 0) or \
             (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
            continue
          trans.putpixel((x, y), pixel)

      if not os.path.exists("out"):
        os.mkdir("out")

      o = Image.open(args.image_path).convert('RGB')
      ow, oh = o.size
      o.save("out/original.jpg")

      trans.save("out/pred.png")
