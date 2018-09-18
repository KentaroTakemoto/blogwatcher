# from PIL import Image
# o = Image.open("out/original.jpg").convert('RGB')


import cv2
o = cv2.imread("out/original.jpg", 1)
ow, oh = o.shape[:2]
p = cv2.imread("out/pred.png", 1)

p = cv2.resize(p, (ow, oh))
pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)

# cv2.imwrite("out/pred_{}.png".format(img_name), pred)
#   os.remove("out/original.jpg")
#   os.remove("out/pred.png")

#   cv2.imshow("image", pred)
#   while cv2.waitKey(33) != 27:
#     pass

cv2.imwrite("out/combined.jpg", pred)
