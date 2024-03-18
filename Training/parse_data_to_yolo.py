import numpy as np
import os
import xml.etree.ElementTree as ET

splitter="/" # changes from windows to linux
label_encode={"receipt":0,
"shop":1,
"total":2,
"item":3,
"date_time":4}



def convert_x1x2y1y2_to_yolo(x1,x2,y1,y2,width,height):
    box_h=(y2-y1)
    box_w=(x2-x1)
    y_center=y1+(box_h/2)
    x_center=x1+(box_w/2)
    return float(x_center/width),float(y_center/height),float(box_w/width),float(box_h/height)
def convert_points_to_yolo(points,width,height):
    points_list=[[float(y[0]) ,float(y[1]) ] for y in [x.split(",") for x in points.split(";") ]]
    points_array=np.array(points_list)
    x1=points_array[:,0].min()
    x2=points_array[:,0].max()
    y1=points_array[:,1].min()
    y2=points_array[:,1].max()
    box_h=(y2-y1)
    box_w=(x2-x1)
    y_center=y1+(box_h/2)
    x_center=x1+(box_w/2)
    return float(x_center/width),float(y_center/height),float(box_w/width),float(box_h/height)

def parse_coordinates(annotation_file,input_folder,output_dir,splitter):
    os.mkdir(output_dir) if not os.path.isdir(output_dir)else""
    os.system("cp "+input_folder+splitter+"*"+" "+output_dir)

    # Load the XML file
    #"ocr-receipts-text-detection"+splitter+"annotations.xml"
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    data_dir=output_dir+splitter
    # Iterate over the images
    points_dict=dict()
    for image in root.findall('image'):
        image_id = image.get('id')
        image_name = image.get('name')
        image_width = float(image.get('width'))
        image_height =float( image.get('height'))

        # Process the image annotations
        lines=[]
        for box in image.findall('box'):
            label = box.get('label')
            xtl=float(box.get('xtl'))
            xbr=float(box.get('xbr'))
            ytl=float(box.get('ytl'))
            ybr=float(box.get('ybr'))
            x,y,h,w=convert_x1x2y1y2_to_yolo(xtl,xbr,ytl,ybr,image_width,image_height)
            label_index=label_encode[label]
            line=" ".join([str(label_index),str(x),str(y),str(h),str(w)])
            lines.append(line)




        for polygon in image.findall('polygon'):
            label = polygon.get('label')
            points=polygon.get('points')
            points_dict[int(image_id)]=points
            x,y,h,w=convert_points_to_yolo(points,image_width,image_height)
            label_index=label_encode[label]
            line=" ".join([str(label_index),str(x),str(y),str(h),str(w)])
            lines.append(line)



        with open(data_dir+image_id+".txt", 'a+') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
    return points_dict
