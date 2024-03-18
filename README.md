# Text Detection using Yolo 

## Inference 
### first we build docker image file 
    `docker build -t detect_text:1 . `
### after that we run the container and we expose our port 8080
    `docker run -p 8080:3000 detect_text bash`
### then use this block of code to simulate the API request :

```
import requests
import base64

# Encode an image to base64 (replace with your actual image)
with open("Inference/1.jpg", "rb") as img_file:
    image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

# Send a POST request to the endpoint
url = "http://localhost:8080/detect_text"  
response = requests.post(url, json={"image_base64": image_base64})

# Print the detected objects (assuming the response is in JSON format)
if response.status_code == 200:
    result = response.json()
    image_base64=result["annotated_base64"]
    boxes=result["result"]


    print(f"Detected texts: {boxes}")
    
else:
    print(f"Error: {response.status_code}, {response.text}")

```
## API Instructions :
* Input format  : {"image_base64": base64 code }
* output format  : {"result":dictionary of label coordinates pair for each label
                    "annotated_base64" : the result image with bounding boxes in base64 format }
