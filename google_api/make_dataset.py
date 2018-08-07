# coding: utf-8
from PIL import Image, ImageDraw
import numpy as np
from pygeocoder import Geocoder
import urllib
import json
import glob

state_names = glob.glob("~/data/jsons//*.json")
for file in state_names:
    state_name = file.split('/')[-1].split('.')[0]
# state_name = 'DistrictofColumbia'
    f = open('{}.json'.format(state_name, 'r')
    json_dict = json.load(f)

    meter_per_pixel = 100 / (np.sqrt(206**2 + 10**2))
    lat_meter = 6378150*2*np.pi/(360)
    current_lat = json_dict["features"][i]["geometry"]["coordinates"][0][0]
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

    def get_pixel(lat, long, center_lat, center_long,img_size=640,pad=0):
        lat_pix = int(img_size/2 -(lat-center_lat)*lat_meter/meter_per_pixel)
        long_pix = int(img_size/2 +(long-center_long)*long_meter/meter_per_pixel)
        return long_pix+pad,lat_pix+pad


    def create_label(center_lat,center_long,directory,image_size=640):
        contain_flag = False
        _,filename = make_url(center_lat,center_long, zoom=18)

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

    pic_directory = "~/data/pictures/{}/".format(state_name)
    lab_directory = "~/data/labels/{}/".format(state_name)
    if not os.path.exists(pic_directory):
        os.mkdir(pic_directory)
    if not os.path.exists(lab_directory):
        os.mkdir(lab_directory)

    if os.path.exists('lat_long_list.npy'):
        lat_long_list = list(np.load('lat_long_list.npy'))
    else:
        lat_long_list = []

    def save(lat_long_list):
        np.save('lat_long_list.npy',np.array(lat_long_list))
        with open("train.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lat_long_list))
        f.close()

    for lat in np.arange(min_lat+margin_lat, max_lat+margin_lat, stride_lat):
        for long in np.arange(min_long+margin_long, max_long+margin_long, stride_long):
            save(lat_long_list)
            url, center = make_url(lat,long, zoom=18)
            if center in lat_long_list:
                continue
            flag = create_label(lat,long,lab_directory,image_size=640)
            if flag:
                download_pic(url,center,pic_directory)
                lat_long_list.append('{}/{}'.format(state_name,center))

    print('data_shape')
    print(np.arange(min_lat+margin_lat, max_lat+margin_lat, stride_lat).shape)
    print(np.arange(min_long+margin_long, max_long+margin_long, stride_long).shape)
    print(len(lat_long_list))
