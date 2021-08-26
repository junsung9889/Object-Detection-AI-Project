from os import listdir, mkdir
from os.path import isfile, join, exists
import shutil

files = [f for f in listdir('../VOC2012/JPEGImages/')]
for i in range(len(files)):
    if i % 10 == 0 or i % 10 == 1:
        continue
    if not exists('../VOC2012/TrainImages'):
        mkdir('../VOC2012/TrainImages')
    shutil.copyfile(f'../VOC2012/JPEGImages/{files[i]}',f'../VOC2012/TrainImages/{files[i]}')

for i in range(len(files)):
    if i % 10 != 0:
        continue
    if not exists('../VOC2012/ValidImages'):
        mkdir('../VOC2012/ValidImages')
    shutil.copyfile(f'../VOC2012/JPEGImages/{files[i]}',f'../VOC2012/ValidImages/{files[i]}')

for i in range(len(files)):
    if i % 10 != 1:
        continue
    if not exists('../VOC2012/TestImages'):
        mkdir('../VOC2012/TestImages')
    shutil.copyfile(f'../VOC2012/JPEGImages/{files[i]}',f'../VOC2012/TestImages/{files[i]}')