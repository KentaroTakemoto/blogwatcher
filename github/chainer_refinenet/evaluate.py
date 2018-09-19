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

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='RefineNet on Chainer (predict)')
  parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
  parser.add_argument('--train_txt', '-tt', default='/home/ppdev/data/train.txt', type=str)
  parser.add_argument('--class_num', '-n', default=21, type=int)
  parser.add_argument('--weight1', '-w1', default="/home/ppdev/data/weights/test3/chainer_refinenet_tmp.weight", type=str)
  # parser.add_argument('--weight2', '-w2', default="/home/ppdev/data/weights/test3/chainer_refinenet_tmp.weight", type=str)
  # parser.add_argument('--weight3', '-w3', default="/home/ppdev/data/weights/test4/chainer_refinenet_tmp.weight", type=str)
  args = parser.parse_args()


  train_txt = args.train_txt
  with open(train_txt,"r") as f:
    ls = f.readlines()
  names = [l.rstrip('\n') for l in ls]
  n_data = len(names)
  np.random.seed(seed=42)
  test_inds = np.random.permutation(n_data)[:100]

  scores = {}
  score_sum = 0
  counter = 0
  for i,test_ind in enumerate(test_inds):
    if i%10==0:
        print(i)
    name = names[test_ind]
    label = np.array(Image.open('/home/ppdev/data/labels/'+name+".png").resize((224,224)))
    if np.isclose(np.sum(label),0):
        score = 1
        counter+=1
        print(name)
        scores[name] = score
        score_sum += score
        continue

    img = Image.open('/home/ppdev/data/pictures/'+name+".png")
    pred1 = predict(img, args.weight1, args.class_num, args.gpu)
    pred1 = chainer.cuda.to_cpu(pred1[0].argmax(axis=0))
    # pred2 = predict(img, args.weight2, args.class_num, args.gpu)
    # pred2 = chainer.cuda.to_cpu(pred2[0].argmax(axis=0))
    # pred3 = predict(img, args.weight3, args.class_num, args.gpu)
    # pred3 = chainer.cuda.to_cpu(pred3[0].argmax(axis=0))

    # pred = np.int8(np.logical_or(pred1,pred2))
    # pred = np.int8(np.logical_or(pred,pred3))
    pred = pred1
    if np.isclose(np.sum(pred),0) and np.isclose(np.sum(label),0):
        score = 1
        counter+=1
        print(name)
    else:
        score = np.sum(np.logical_and(pred, label)) / np.sum(np.logical_or(pred, label),dtype=np.float32)
    scores[name] = score
    score_sum += score
  print(counter)
  print((score_sum-counter)/(100-counter))
  import pickle
  with open('/home/ppdev/codes/blogwatcher/github/chainer_refinenet/results/test3.pickle', 'wb') as f:
      pickle.dump(scores, f)
