import numpy
from PIL import Image
import glob
import os;
import re
import csv


def prepare_img(path):
    with open('image.csv') as csv_file:
        count = 0;
        names =[]
        labels = []
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        readCSV = csv.reader(csv_file, delimiter = ',')
        for row in readCSV:
            if count == 0:
                count = count + 1
                continue #skip the first row of the csv file
            name = row[0]
            label = row[3]
            xmin = row[4]
            ymin = row[5]
            xmax = row[6]
            ymax = row[7]

            names.append(name)
            labels.append(label)
            xmins.append(xmin)
            ymins.append(ymin)
            xmaxs.append(xmax)
            ymaxs.append(ymax)

    offset = 0
    buffer = 0
    for png in glob.glob(path + '\*.jpg'):
        for i in range(3):
            img = Image.open(png)
            area = (int(xmins[i+offset]), int(ymins[i+offset]), int(xmaxs[i+offset]), int(ymaxs[i+offset]))
            img=img.convert('L')
            img = img.crop(area)
            img = img.resize((28,28))

            if offset % 24 == 0:
                img.save("test_img\\" + labels[i + offset] + "_" + str(buffer) + ".png")
            else:
                img.save("Procced_img\\" + labels[i + offset] + "_" + str(buffer) + ".png")
            buffer = buffer + 1
        offset = offset + 3





#main starts here

path = r"Project1_Raw_Data"

prepare_img(path)
