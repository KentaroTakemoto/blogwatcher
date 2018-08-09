# coding: utf-8
from PIL import Image, ImageDraw
import numpy as np
from pygeocoder import Geocoder
import urllib
import json
import glob
import os
import time

def make_url(lat,long,key,zoom=18):
    html1 = "https://maps.googleapis.com/maps/api/staticmap?center="
    html2 = "&maptype=satellite&size=640x640&sensor=false&zoom="
    html3 = "&key="
    center = str(lat) + "," + str(long)
    html = html1 + center + html2 + str(zoom) + html3 + key
    return html, center

def download_pic(url,filename,directory,key_num):
    try:
        with urllib.request.urlopen(url) as url_:
            img = url_.read()
    except urllib.error.HTTPError as e:
        print(e.code)
        print(e.read())
        img = False
        key_num += 1
    if img:
        localfile = open(directory+str(filename)+".png",'wb')
        localfile.write(img)
        localfile.close()
    return key_num

def get_pixel(lat, long, center_lat, center_long,img_size=640,pad=0):
    lat_pix = int(img_size/2 -(lat-center_lat)*lat_meter/meter_per_pixel)
    long_pix = int(img_size/2 +(long-center_long)*long_meter/meter_per_pixel)
    return long_pix+pad,lat_pix+pad


def create_label(center_lat,center_long,directory,image_size=640):
    contain_flag = False
    filename = str(center_lat) + "," + str(center_long)

    buildings = []
    margin = int(image_size/2)
    for i in range(len(json_dict["features"])):
        coordinates = json_dict["features"][i]["geometry"]["coordinates"][0]
        flag = 0
        for coordinate in coordinates:
            long,lat = get_pixel(coordinate[1], coordinate[0], center_lat, center_long,img_size=image_size)
            if -margin< lat and lat<image_size+margin and -margin<long and long<image_size+margin:
                flag+=1
        if flag >= len(coordinates):
            buildings.append(coordinates)
            contain_flag = True

    if not contain_flag:
        return False

    image = Image.new('P', (image_size*2,image_size*2),(0)) # 画面外にはみ出る建物にも対応するため余白をとる
    draw = ImageDraw.Draw(image)
    for building in buildings:
        pix_building = [get_pixel(long_lat[1],long_lat[0], center_lat, center_long,pad=margin) for long_lat in building]
        draw.polygon(tuple(pix_building), fill=(1), outline=(255))

    Image.fromarray(np.asarray(image)[margin:margin+image_size,margin:margin+image_size]).save(directory+filename+".png")
    return True

def save(lat_long_list):
    np.save('lat_long_list.npy',np.array(lat_long_list))
    with open("train.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lat_long_list))
    f.close()

api_keys = ["AIzaSyACqUbsT9E2eShglWPm01btIMKjBQQD948", "AIzaSyCh1nSA01a_9LzvHOKHsFuP5CZLauzcpfI"]
key_num = 0
pre_key_num = 0

state_names = glob.glob("/home/ppdev/data/jsons/*.json")
for file in state_names:
    state_name = file.split('/')[-1].split('.')[0]
    print('------------------------')
    from datetime import datetime
    datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(state_name)
# state_name = 'DistrictofColumbia'
    f = open('/home/ppdev/data/jsons/{}.json'.format(state_name, 'r'))
    json_dict = json.load(f)

    meter_per_pixel = 100 / (np.sqrt(206**2 + 10**2))
    lat_meter = 6378150*2*np.pi/(360)
    # print(json_dict["features"][0]["geometry"]["coordinates"][0])
    # print(json_dict["features"][0]["geometry"]["coordinates"][0][0])
    current_lat = float(json_dict["features"][0]["geometry"]["coordinates"][0][0][1])
    long_meter = 6378150*np.cos(current_lat/180*np.pi)*2*np.pi/(360)

    min_long = min_lat = np.inf
    max_long = max_lat = -np.inf
    for i in range(len(json_dict["features"])):
        coordinates = json_dict["features"][i]["geometry"]["coordinates"][0]
        for coordinate in coordinates:
            if coordinate[0]<min_long:
                min_long = coordinate[0]
            if coordinate[1]<min_lat:
                min_lat = coordinate[1]
            if coordinate[0]>max_long:
                max_long = coordinate[0]
            if coordinate[1]>max_lat:
                max_lat = coordinate[1]

    print('min_lat, max_lat, min_long, max_long:')
    print(min_lat, max_lat, min_long, max_long)

    margin_lat = meter_per_pixel*320/lat_meter
    stride_lat = meter_per_pixel*640/lat_meter
    margin_long = meter_per_pixel*320/long_meter
    stride_long = meter_per_pixel*640/long_meter

    pic_directory = "/home/ppdev/data/pictures/{}/".format(state_name)
    lab_directory = "/home/ppdev/data/labels/{}/".format(state_name)
    if not os.path.exists(pic_directory):
        os.mkdir(pic_directory)
    if not os.path.exists(lab_directory):
        os.mkdir(lab_directory)

    if os.path.exists('lat_long_list.npy'):
        lat_long_list = list(np.load('lat_long_list.npy'))
    else:
        lat_long_list = []



    from datetime import datetime
    datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print("start download...")
    count = 0
    for lat in np.arange(min_lat+margin_lat, max_lat+margin_lat, stride_lat):
        print("lat:{}".format(lat))
        if lat==39.7212756378488:
            continue
        for long in np.arange(min_long+margin_long, max_long+margin_long, stride_long):
            print("long:{}".format(long))
            url, center = make_url(lat,long,api_keys[key_num], zoom=18)
            if '{}/{}'.format(state_name,center) not in lat_long_list:
                if count==1:
                    start = time.time()
                    print("start time")
                elif count==2:
                    elapsed_time = time.time() - start
                    print ("elapsed_time1:{0}".format(elapsed_time) + "[sec]")
                elif count==11:
                    elapsed_time = time.time() - start
                    print ("elapsed_time10:{0}".format(elapsed_time) + "[sec]")
                elif count==101:
                    elapsed_time = time.time() - start
                    print ("elapsed_time100:{0}".format(elapsed_time) + "[sec]")

                save(lat_long_list)
                flag = create_label(lat,long,lab_directory,image_size=640)
                if flag:
                    count += 1
                    key_num = download_pic(url,center,pic_directory,key_num)
                    if pre_key_num!=key_num:
                        url, center = make_url(lat,long,api_keys[key_num], zoom=18)
                        key_num = download_pic(url,center,pic_directory,key_num)
                        pre_key_num = key_num
                    lat_long_list.append('{}/{}'.format(state_name,center))
                if len(lat_long_list)%100==0:
                    print(len(lat_long_list))
        print("len(lat_long_list):{}".format(len(lat_long_list)))

    print('data_shape')
    print(np.arange(min_lat+margin_lat, max_lat+margin_lat, stride_lat).shape)
    print(np.arange(min_long+margin_long, max_long+margin_long, stride_long).shape)
    print(len(lat_long_list))
    from datetime import datetime
    datetime.now().strftime("%Y/%m/%d %H:%M:%S")
