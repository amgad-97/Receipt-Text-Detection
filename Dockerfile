
FROM python:3.10

WORKDIR /app

COPY ./Inference .
COPY ./requirements.txt .


RUN pip install --no-cache-dir -r ./requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN gdown https://drive.google.com/uc?id=1-28anYCBa8Cs0mqYE3EqqwPJ7SNZSFtJ


# OK, now we pip install our requirements

EXPOSE 3000

CMD uvicorn main:app --host 0.0.0.0 --port 3000
