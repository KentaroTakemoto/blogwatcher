import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable
from chainer import training
from chainer.training import extensions

import sys
import os
import argparse
import random

from refinenet import RefineResNet, TestNet
from resnet101 import ResNet
from preprocess import load_data

def copy_model(src, dst):
  assert isinstance(src, chainer.Chain)
  assert isinstance(dst, chainer.Chain)
  for child in src.children():
    if child.name not in dst.__dict__: continue
    dst_child = dst[child.name]
    if type(child) != type(dst_child): continue
    if isinstance(child, chainer.Chain):
      copy_model(child, dst_child)
    if isinstance(child, chainer.Link):
      match = True
      for a, b in zip(child.namedparams(), dst_child.namedparams()):
        if a[0] != b[0]:
          match = False
          break
        if a[1].data.shape != b[1].data.shape:
          match = False
          break
      if not match:
        print('Ignore %s because of parameter mismatch' % child.name)
        continue
      for a, b in zip(child.namedparams(), dst_child.namedparams()):
        b[1].data = a[1].data
      # print('Copy %s' % child.name)

# entry point
parser = argparse.ArgumentParser(description='RefineNet on Chainer (train)')
parser.add_argument('--gpu', '-g', default=0, type=int,
          help='GPU ID (negative value indicates CPU)')
# parser.add_argument('--train_dataset', '-tr', default='dataset', type=str)
# parser.add_argument('--target_dataset', '-ta', default='dataset', type=str)
parser.add_argument('--train_txt', '-tt', default='/home/ppdev/data/train.txt', type=str)
parser.add_argument('--batchsize', '-b', type=int, default=10,
          help='batch size (default value is 1)')
parser.add_argument('--initmodel', '-i', default=None, type=str,
          help='initialize the model from given file')
parser.add_argument('--epoch', '-e', default=50, type=int)
parser.add_argument('--class_num', '-n', default=21, type=int)
parser.add_argument('--lr', '-l', default=1e-4, type=float)
args = parser.parse_args()

n_epoch = args.epoch
batchsize = args.batchsize
# train_dataset = args.train_dataset
# target_dataset = args.target_dataset
train_dataset = '/home/ppdev/data/pictures/'
target_dataset = '/home/ppdev/data/labels/'
weight_path = '/home/ppdev/data/weights/test1/'
if not os.path.exists(weight_path):
    od.mkdir(weight_path)

train_txt = args.train_txt

with open(train_txt,"r") as f:
  ls = f.readlines()
names = [l.rstrip('\n') for l in ls]
n_data = len(names)
n_iter = n_data // batchsize
gpu_flag = True if args.gpu > 0 else False

model = RefineResNet(args.class_num)
# model = TestNet(args.class_num)

if args.initmodel:
  serializers.load_npz(args.initmodel, model)
  print("Load initial weight")
else:
  resnet = ResNet()
  # serializers.load_npz("resnet101.npz", resnet)
#   caffemodel = '/data/unagi0/takemoto/blogwatcher/weights/ResNet-101-model.caffemodel'
#   chainermodel = '/data/unagi0/takemoto/blogwatcher/weights/ResNet-101-model.npz'
#   L.ResNet101Layers.convert_caffemodel_to_npz(caffemodel, chainermodel, n_layers = 101)
#   serializers.load_npz(chainermodel, resnet)
  copy_model(resnet, model)

if args.gpu >= 0:
  chainer.cuda.get_device(args.gpu).use()
  model.to_gpu()

xp = np if args.gpu < 0 else cuda.cupy

optimizer = optimizers.Adam(alpha=args.lr)

optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5), 'hook_fcn')

print("## INFORMATION ##")
print("Num Data: {}, Batchsize: {}, Iteration {}".format(n_data, batchsize, n_iter))


print("-"*40)
loss_history = []
best_epoch = 0
best_loss = np.inf
for epoch in range(1, n_epoch+1):
  print('epoch', epoch)
  random.shuffle(names)

  if epoch == 3:
    optimizer.alpha *= 0.1
  elif epoch == 6:
    optimizer.alpha *= 0.1
  elif epoch % 10 == 0:
    optimizer.alpha *= 0.1

  loss_sum = 0
  for i in range(n_iter):

    model.zerograds()
    indices = range(i * batchsize, (i+1) * batchsize)

    x = xp.zeros((batchsize, 3, 224, 224), dtype=np.float32)
    y = xp.zeros((batchsize, 224, 224), dtype=np.int32)
    for j in range(batchsize):
      name = names[i*batchsize + j]
      xpath = train_dataset+name+".png"
      ypath = target_dataset+name+".png"

      if random.randint(0, 1):
        hflip = True
      else:
        hflip = False

      rs = random.randint(256, 480)
      xs = random.randint(0, rs-225)
      ys = random.randint(0, rs-225)
      x[j] = load_data(xpath, crop=True, mode="data", hflip=hflip, rcrop=True, xs=xs, ys=ys, rs=rs, xp=xp)
      if x[j] is not None:
          y[j] = load_data(ypath, crop=True, mode="label", hflip=hflip, rcrop=True, xs=xs, ys=ys, rs=rs, xp=xp)
      else:
          x[j] = x[0]
          y[j] = y[0]

    x = Variable(x)
    y = Variable(y)
    with chainer.using_config('train', True):
      loss = model(x, y)

    if i%10==0:
        sys.stdout.write("\r%s" % "batch: {}/{}, loss: {}".format(i+1, n_iter, loss.data))
        sys.stdout.flush()

    loss.backward()
    optimizer.update()
    loss_sum += loss.data

  loss = loss_sum/n_iter
  print("\n average loss: ", loss)
  print("-"*40)
  if best_loss>loss:
    fn = weight_path+'chainer_refinenet_tmp.weight'
    serializers.save_npz(fn, model)
    best_epoch = epoch
    best_loss = loss

  print("best_epoch:{} best_loss:{}".format(best_epoch,best_loss))
  loss_history.append(loss)
  print("history:")
  print(loss_history)


serializers.save_npz(weight_path+'chainer_refinenet_final.weight', model)
serializers.save_npz(weight_path+'chainer_refinenet_final.state', optimizer)
