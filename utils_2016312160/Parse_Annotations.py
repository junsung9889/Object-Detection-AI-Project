import xml.etree.ElementTree as ET
import os

classes = ( 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')

def parseAnnotation(filename):
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        dic = {}
        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue
        dic['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        dic['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(dic)
    return objects


txt_file = open('../voc2012_train.txt','w')

count = 0
for annotation in os.listdir('../VOC2012/Annotations/'):
    # For train set
    if count % 10 == 0 or count % 10 == 1:
        count += 1
        continue

    count += 1

    image_path = annotation.split('.')[0] + '.jpg'
    results = parseAnnotation('../VOC2012/Annotations/' + annotation)
    if len(results)==0:
        continue
    txt_file.write(image_path)
    for result in results:
        class_name = result['name']
        bbox = result['bbox']
        class_name = classes.index(class_name)
        txt_file.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name))
    txt_file.write('\n')
    
txt_file.close()

txt_file = open('../voc2012_valid.txt','w')

count = 0
for annotation in os.listdir('../VOC2012/Annotations/'):

    #  For valid set
    if count % 10 != 0:
        count += 1
        continue
    count += 1

    image_path = annotation.split('.')[0] + '.jpg'
    results = parseAnnotation('../VOC2012/Annotations/' + annotation)
    if len(results)==0:
        continue
    txt_file.write(image_path)
    for result in results:
        class_name = result['name']
        bbox = result['bbox']
        class_name = classes.index(class_name)
        txt_file.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name))
    txt_file.write('\n')
    
txt_file.close()

txt_file = open('../voc2012_test.txt','w')

count = 0
for annotation in os.listdir('../VOC2012/Annotations/'):

    # For test set
    if count % 10 != 1:
        count += 1
        continue

    count += 1

    image_path = annotation.split('.')[0] + '.jpg'
    results = parseAnnotation('../VOC2012/Annotations/' + annotation)
    if len(results)==0:
        continue
    txt_file.write(image_path)
    for result in results:
        class_name = result['name']
        bbox = result['bbox']
        class_name = classes.index(class_name)
        txt_file.write(' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])+' '+str(class_name))
    txt_file.write('\n')
    
txt_file.close()