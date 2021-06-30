import os
import logging
import datetime
from logging import Formatter, FileHandler
from flask import Flask, request, jsonify, render_template

from east_pytes import process_image, _get_image, convert_jpg, is_url_image, _get_image_dir

_VERSION = 1  # API version

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/v{}/ocr'.format(_VERSION), methods=["POST"])
def ocr():
    start = datetime.datetime.now()
    try:
        url = request.json['image_url']
        if is_url_image(url):
            if 'jpg' in url:
                image = _get_image(url)
                print("hi")
                text = process_image(image)
                with open("log_file.txt", "a") as log:
                    log.write(str(start) + " | TEXT: " + text + "\n\n")
                return jsonify({"output": text})
        else:
            image = _get_image(url)
            image = convert_jpg(image)
            text = process_image(image)
            with open("log_file.txt", "a") as log:
                log.write(str(start) + " | TEXT: " + text + "\n\n")
            return jsonify({"output": text})

    except:
        return jsonify(
            {"error": "Please enter a valid URL!"}
        )

@app.route('/', methods=["POST"])
def upload_img():
    if request.method == 'POST':
        start = datetime.datetime.now()
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('index.html', msg='No file selected')
        file = request.files['file']
        # if no file is selected
        if file.filename == '':
            return render_template('index.html', msg='No file selected')

        if file and allowed_file(file.filename):

            # call the OCR function on it
            file.save(UPLOAD_FOLDER + file.filename)
            image = _get_image_dir(UPLOAD_FOLDER + file.filename)
            image = convert_jpg(image)
            extracted_text = process_image(image)
            with open("log_file.txt", "a") as log:
                log.write(str(start) + " | TEXT: " + extracted_text + "\n\n")
            # extract the text and display it
            return render_template('upload.html',
                                   extracted_text=extracted_text,
                                   img_src=UPLOAD_FOLDER + file.filename)
    elif request.method == 'GET':
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')