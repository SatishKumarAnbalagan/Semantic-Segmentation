#!/usr/bin/python

import glob
from operator import itemgetter
from os import walk


outputfile = "./val.txt"  # file to save the results to
folders = ['./input/leftImg8bit_trainvaltest/val/', './input/gtFine_trainvaltest/val/', './input/hed_gtFine_trainvaltest/val/']

f = []

with open(outputfile, "w") as txtfile:
    for folder in folders:
        fileList = []
        for (dirpath, dirnames, filenames) in walk(folder):
            for files in filenames:
                #txtfile.write("%s\n" % (str(dirpath) + str(files)))
                fileList.append("%s" % str(dirpath) + str(files))
        f.append(fileList)
        #txtfile.write("\n" + str(f))

    p = 0
    for i in f[0]:
        print(p)
        txtfile.write("%s\n" % ', '.join(map(itemgetter(p), f)))
        p = p + 1

txtfile.close()
