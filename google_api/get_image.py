from PIL import Image, ImageDraw
import numpy as np
from pygeocoder import Geocoder
import urllib
import json

def download_pic(url,filename,directory):
    with urllib.request.urlopen(url) as url_:
        img = url_.read()
    localfile = open(directory+str(filename)+".png",'wb')
    localfile.write(img)
    localfile.close()

def make_url(lat,long, zoom=18, key='AIzaSyCh1nSA01a_9LzvHOKHsFuP5CZLauzcpfI'):
    html1 = "https://maps.googleapis.com/maps/api/staticmap?center="
    html2 = "&maptype=satellite&size=640x640&sensor=false&zoom="
    html3 = "&key="
    center = str(lat) + "," + str(long)
    html = html1 + center + html2 + str(zoom) + html3 + key
    return html, center

# # ローカルに保存するときはこっち
# def download_pic_(url,filename):
#     with urllib.request.urlopen(url) as url_:
#         img = url_.read()
# # 	img = urllib.urlopen(url)
#     localfile = open("./"+str(filename)+".png",'wb')
#     localfile.write(img)
# #     img.close()
#     localfile.close()

directory = "/home/ppdev/code/blogwatcher/github/chainer_refinenet/test_images/"
if not os.path.exists(directory):
    os.mkdir(directory)

lat_long_list = []
lat = 35.6776893
long = 139.7768978 - (0.00358696351780452*2)
for i in range(5):
    url, center = make_url(lat,long, zoom=18)
    if center in lat_long_list:
        continue
    download_pic(url,center,directory)
    lat_long_list.append(center)
    long += 0.00358696351780452
