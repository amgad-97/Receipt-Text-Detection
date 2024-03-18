# Inference for ONNX model
import cv2
import glob
import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
import torch
import base64

try :   
    cuda = True if torch.cuda.is_available() else False
except:
    cuda=False
w = "model.onnx"

label_encode={0:"receipt",
1:"shop",
2:"total",
3:"item",
4:"date_time"}
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)


def image_to_base64(image_path):

  with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()
  return base64.b64encode(image_bytes).decode("utf-8")


def base64_to_cv2(base64_str):
    # Decode the base64 string into a numpy array
    np_arr = np.frombuffer(base64.b64decode(base64_str), dtype=np.uint8)

    # Decode the numpy array into an image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    print(img)

    return img

def post_process(ll):
  if ll:
    list_array=np.array(ll)
    arr=list_array.max(axis=0)
    return [arr.tolist()]
  else:
    return ll




def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)



def onnx_inference(img):
  result={"receipt":[],
      "shop":[],
      "total":[],
      "item":[],
      "date_time":[]}
  image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  image, ratio, dwdh = letterbox(image, auto=False)
  image = image.transpose((2, 0, 1))
  image = np.expand_dims(image, 0)
  image = np.ascontiguousarray(image)
  im = image.astype(np.float32)
  im /= 255
  outname = [i.name for i in session.get_outputs()]
  inname = [i.name for i in session.get_inputs()]
  inp = {inname[0]:im}
  outputs = session.run(outname, inp)[0]
  for output in outputs:
      batch_id,x0,y0,x1,y1,cls_id,score=output
      box = np.array([x0,y0,x1,y1])
      box -= np.array(dwdh*2)
      box /= ratio
      box = box.round().astype(np.int32).tolist()
      x0,y0,x1,y1=box
      score = round(float(score),3)
      name = label_encode[cls_id]
      result[name].append([x0,y0,x1,y1])
  one_result=["receipt","shop","total","date_time"]
  for k in one_result :
    result[k]=post_process( result[k])
  return result

def draw_boxes(img,result):

  names = result.keys()
  colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}

  for label,boxes in result.items():
    if boxes:
      for box in boxes:
        x0,y0,x1,y1=box
        color=colors[label]
        cv2.rectangle(img,box[:2],box[2:],color,2)
        cv2.putText(img,label,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,color,thickness=2)
  return img

if __name__ == "__main__" :

    images_path="./*.jpg"
    output_folder_annotation="./"
    imgs_list=glob.glob(images_path)
    for img_name in imgs_list :
        base64_im=image_to_base64(img_name)
        img=base64_to_cv2(base64_im)
        result=onnx_inference(img)
        image=draw_boxes(img,result)
        cv2.imwrite("./"+(output_folder_annotation+img_name.split("/")[-1]),image)
