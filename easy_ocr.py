import io

import easyocr
import cv2
import os

import requests
import torch
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFilter
from io import StringIO
from io import BytesIO



def process_image(image):
    reader = easyocr.Reader(['en', 'tr'])
    filename = "img.jpg"
    res = reader.readtext(image)
    print(res)

    txt = ""

    #fontsize = 10
    #font = ImageFont.truetype("arial.ttf", fontsize)

    for (bb, text, prob) in res:
        # Get the bounding box
        (x1, y1, x2, y2) = bb
        x1 = (int(x1[0]), int(x1[1]))
        y1 = (int(y1[0]), int(y1[1]))
        x2 = (int(x2[0]), int(x2[1]))
        y2 = (int(y2[0]), int(y2[1]))

        txt = txt + " " + text + " "
        shape = [x1, x2]
        print(shape)
        # Draw rectangle
        # img = ImageDraw.Draw(image)
        # img.rectangle(shape, outline="red")

        #Put text
        # img.text(x1, text, fill="red", font=font, align ="left")

    # image.show()
    # image.save("C:/Users/MELİH/IdeaProjects/ocr_new/static/" + filename)
    return txt

def _get_image(url):
    return Image.open(BytesIO(requests.get(url).content))

def _get_image_dir(path):
    return Image.open(path)


def convert_jpg(img):
    if not img.mode == 'RGB':
        img = img.convert('RGB')
    with BytesIO() as f:
        img.save(f, format='JPEG')
        return f.getvalue()

def is_url_image(url):
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    r = requests.head(url)
    if r.headers["content-type"] in image_formats:
        return True
    return False
