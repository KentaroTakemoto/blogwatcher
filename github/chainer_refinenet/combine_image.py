import cv2
o = cv2.imread("out/original.jpg", 1)
ow, oh = o.shape[:2]
p = cv2.imread("out/pred.png", 1)

p = cv2.resize(p, (ow, oh))
pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)

cv2.imwrite("out/combined.jpg", pred)

# with open("github/chainer_refinenet/test_images/japan5.txt","r") as f:
#   ls = f.readlines()
# names = [l.rstrip('\n') for l in ls]
#
# import cv2
# for name in names:
#     o = cv2.imread("github/chainer_refinenet/test_images/{}.png".format(name), 1)
#     ow, oh = o.shape[:2]
#     p = cv2.imread("github/chainer_refinenet/test_images/pred_{}.png".format(name), 1)
#
#     p = cv2.resize(p, (ow, oh))
#     pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)
#
#     cv2.imwrite("github/chainer_refinenet/test_images/combined_{}.png".format(name), pred)
