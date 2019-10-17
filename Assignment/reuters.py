#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 11:32:52 2017

@author: AAdhikari
"""
import os
import csv

def build_data_frame(rootdir,data_type):
    i=0
    data_frame = []
    for root, subFolders, files in os.walk(rootdir):
        if(i==0):
            subfold = subFolders       
        if(i>0):
            for filename in files: 
                fin = open(os.path.join(root,filename), 'r')
                data=fin.read().replace('\n', '')
                data_frame_row = []
                data_frame_row.append(data)
                data_frame_row.append(subfold[i-1])
                data_frame_row.append(data_type)
                data_frame.append(data_frame_row)
                data = ""
                fin.close()
        i=i+1
    return data_frame

rootdir = "C:\Users\Machine_Learning_Assignments\Assignment_1\reuters_21578_10cat\training"
reuters_train = build_data_frame(rootdir,'train')
fout = open("C:\Users\Machine_Learning_Assignments\Assignment_1\reuters_train.csv","wb")
writer = csv.writer(fout)
writer.writerows(reuters_train)
fout.close()

rootdir = "C:\Users\Machine_Learning_Assignments\Assignment_1\reuters_21578_10cat\test"
reuters_test = build_data_frame(rootdir,'test')
fout = open("C:\Users\Machine_Learning_Assignments\Assignment_1\reuters_test.csv","wb")
writer = csv.writer(fout)
writer.writerows(reuters_test)
fout.close()
