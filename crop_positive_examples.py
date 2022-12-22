import cv2 as cv
import os

for char in ['andy', 'louie', 'ora', 'tommy']:
    print(f'Start {char} cropping')
    dir_path = './antrenare/' + char + '/'
    txt_path = './antrenare/' + char + '_annotations.txt'

    f = open(txt_path, 'r')

    coords = {}
    for line in f:
        words = line.split(' ')

        if words[0] not in coords.keys():
            coords[words[0]] = []

        if words[5] in [char, char+'\n']:
            coords[words[0]].append(int(words[1]))
            coords[words[0]].append(int(words[2]))
            coords[words[0]].append(int(words[3]))
            coords[words[0]].append(int(words[4]))
        

    f.close()

    i = 0
    for (key, val) in coords.items():
        if len(val) > 0:
            print(i)
            image = cv.imread(dir_path + key)

            patch = image[val[1]:val[3], val[0]:val[2]]
            patch = cv.resize(patch, (36, 36))

            cv.imwrite('./antrenare/' + char + '_patch/' + str(i) + '.jpg', patch)
            i+=1
    