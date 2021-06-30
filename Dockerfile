FROM python:3.9

WORKDIR /app
COPY . /app


RUN pip install -r requirements.txt
RUN pip install git+git://github.com/jaidedai/easyocr.git
RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]

