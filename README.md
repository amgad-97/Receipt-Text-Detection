# Text Detection using Yolo 

## Inference 
### first we build docker image file 
    `docker build -t detect_text:1 . `
### after that we run the container and we expose our port 8080
    `docker run -p 8080:3000 detect_text bash`