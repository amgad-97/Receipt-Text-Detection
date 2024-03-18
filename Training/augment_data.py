import cv2
import numpy as np
import glob
import os
import tqdm
import xml.etree.ElementTree as ET
import parse_data_to_yolo

splitter="/" # changes from windows to linux
label_encode={"receipt":0,
"shop":1,
"total":2,
"item":3,
"date_time":4}


data_folder="data"
downloaded_dir="ocr-receipts-text-detection"


points_dict=parse_data_to_yolo.parse_coordinates(downloaded_dir+splitter+"annotations.xml",downloaded_dir+splitter+"images",data_folder,splitter=splitter)
def save_annotations(image_object,text_file_source,folder,name,splitter):
    """ Function to save image and text pairs while augmenting.

    Args:
    image_object: the object for the image.
    text_file_source: the source text file.
    folder: folder in which the data will be moved/created.
    splitter: os splitter .
    Returns:
    None
    """

    folder=folder+splitter if folder[-1]!=splitter else folder
    cv2.imwrite(folder+name+".jpg",image_object)
    with open(text_file_source, 'r') as source_file, open(folder+name+".txt", 'a') as dest_file:
        for line in source_file:
            dest_file.write(line)

def augment_pixels(image):
    """ Function that chanes the pixel values of an image based on different aspects.
    1- it create more sharped version of the original image 
    2- it create more blured version of the original image 
    3- it create histogram equlized image (CLAHE) 
    4- it increases the brightness of the original image 
    5- decreases the brightness of the original image 

    Args:
    image: the object for the image.
    Returns:
    augmented_dict : a dictionary of the original image and augmentations with it' corresponding names 
    """

    c1 = cv2.createCLAHE(clipLimit=1, tileGridSize=(8, 8))
    kernel = np.array([[0, -1, 0],
                    [-1, 5,-1],
                    [0, -1, 0]])
    augmented_dict=dict()
    # SHARP
    sharped = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    # BLUR
    blured = cv2.GaussianBlur(image, (5,5), 0)
    # CLAHE
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = c1.apply(gray_image)
    clahe=cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    # BRIGHT
    bright = cv2.add(image, 70)
    # DARK
    dark = cv2.add(image, -70)

    augmented_dict["org"]=image
    augmented_dict["sharp"]=sharped
    augmented_dict["blur"]=blured
    augmented_dict["clahe"]=clahe
    augmented_dict["bright"]=bright
    augmented_dict["dark"]=dark
    return augmented_dict

def augment_data(image_path,output_folder,mask_poiints,pre="_",splitter="/",step=70):

    """ Function to augment the data first by changing the background with different colors 
    and for each color it will augments it's pixels into 5 augmentations .

    Args:
        image_path: path to the original image .
        output_folder:the output folder for the new augmented data
        mask_poiints:the points of the reciept itself to change it's background
        pre: text to write before saving the image/text
        splitter: os splitter
        step : number to control the stride of color range (the larger it gets the small range we got )

    Returns:
        None.
    """

    folder=splitter.join(image_path.split(splitter)[:-1])
    name_=image_path.split(splitter)[-1].split(".")[0]
    image_file_name=image_path.split(splitter)[-1]
    text_file_name=name_+".txt"
    text_path=folder+splitter+name_+".txt"
    image = cv2.imread(image_path)
    foreground_borders = np.array(mask_poiints, dtype=np.int32)
    augmented_dict=augment_pixels(image)
    for aug_type,augmented in augmented_dict.items():
        save_annotations(augmented,text_path,output_folder,"org_"+aug_type+"_"+name_,splitter)

    # Create an empty mask with the same dimensions as the image
    mask = np.zeros_like(image[:, :, 0])

    # Fill the mask with white (255) inside the polygon defined by foreground borders
    cv2.fillPoly(mask, [foreground_borders], color=(255, 255, 255))

    # Invert the mask
    mask1 = cv2.bitwise_not(mask)
    colors=[(x,y,z) for x in range(0,256,step) for y in range(0,256,step) for z  in range(0,256,step)]
    # Create a new background image with the desired color (e.g., blue)
    count=0
    for background_color in colors:

        background = np.full_like(image, background_color)

        # # Get the masked foreground borders
        foreground_masked = cv2.bitwise_and(image, image, mask=mask)

        # # Get the masked background
        background_masked = cv2.bitwise_and(background, background, mask=mask1)

        # Combine the masked foreground borders and masked background
        result = cv2.bitwise_or(foreground_masked, background_masked)
        augmented_dict=augment_pixels(result)
        for aug_type,augmented in augmented_dict.items():
            save_annotations(augmented,text_path,output_folder,pre+"colored_"+str(count)+"_"+aug_type+"_"+name_,splitter)
            count=count+1

#create splitted dataset
os.mkdir("data_splitted") if not os.path.isdir("data_splitted") else "" 
os.mkdir("data_splitted"+splitter+"train") if not os.path.isdir("data_splitted"+splitter+"train") else "" 
os.mkdir("data_splitted"+splitter+"test")if not os.path.isdir("data_splitted"+splitter+"test") else "" 
os.mkdir("data_splitted"+splitter+"val")if not os.path.isdir("data_splitted"+splitter+"val") else "" 

train_folder="data_splitted"+splitter+"train"
val_folder="data_splitted"+splitter+"val"
test_folder="data_splitted"+splitter+"test"

#sassving classes files 
data_dir="data_splitted"+splitter+"val"+splitter
with open(data_dir+"classes.txt", 'a+') as f:
    for label , key in label_encode.items():
        stri=str(key)+" "+label
        f.write(stri)
        f.write('\n')

data_dir="data_splitted"+splitter+"test"+splitter
with open(data_dir+"classes.txt", 'a+') as f:
    for label , key in label_encode.items():
        stri=str(key)+" "+label
        f.write(stri)
        f.write('\n')
data_dir="data_splitted"+splitter+"train"+splitter
with open(data_dir+"classes.txt", 'a+') as f:
    for label , key in label_encode.items():
        print(key , label)
        stri=str(key)+" "+label
        f.write(stri)
        f.write('\n')

#split data from the original 
image_list=glob.glob(data_folder+splitter+"*.jpg")
train_list=image_list[:-4]
val_list=train_list[-4:-2]
test_list=train_list[-2:]
splitter="\\"
print("Augmenting Train Split")
for img in tqdm.tqdm(train_list):
    im_id=int(img.split(splitter)[-1].split(".")[0])
    points=points_dict[im_id]
    points_list=[[int(float(y[0])) ,int(float(y[1])) ] for y in [x.split(",") for x in points.split(";") ]]
    augment_data(img,train_folder,points_list,splitter=splitter)
print("Augmenting Val Split")
for img in tqdm.tqdm(val_list):
    im_id=int(img.split(splitter)[-1].split(".")[0])
    points=points_dict[im_id]
    points_list=[[int(float(y[0])) ,int(float(y[1])) ] for y in [x.split(",") for x in points.split(";") ]]
    augment_data(img,val_folder,points_list,splitter=splitter)
print("Augmenting Test Split")
for img in tqdm.tqdm(test_list):
    im_id=int(img.split(splitter)[-1].split(".")[0])
    points=points_dict[im_id]
    points_list=[[int(float(y[0])) ,int(float(y[1])) ] for y in [x.split(",") for x in points.split(";") ]]
    augment_data(img,test_folder,points_list,splitter=splitter)