import os
import urllib
from io import BytesIO

import imutils
from skimage import io

import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image
from imutils.object_detection import non_max_suppression


def most_likely(scores, char_set):
    text = ""
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        text += char_set[c]
    return text


def map_rule(text):
    char_l = []
    for i in range(len(text)):
        if i == 0:
            if text[i] != '-':
                char_l.append(text[i])
        else:
            if text[i] != '-' and (not (text[i] == text[i - 1])):
                char_l.append(text[i])

    return ''.join(char_l)


def best_outcome(scores, char_set):
    text = most_likely(scores, char_set)
    output = map_rule(text)
    return output


def doOverlap(x11, y11, x12, y12, x21, y21, x22, y22):
    # To check if either rectangle is actually a line
    # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}

    if (x11 == x12 or y11 == y22 or x21 == x22 or y21 == y22):
        # the line cannot have positive overlap
        return False

    # If one rectangle is on left side of other
    if (x11 >= x22 or x21 >= x12):
        return False

    # If one rectangle is above other
    if (y11 >= y22 or y21 >= y12):
        return False

    return True


def mergeBoxes(bb):
    rectsUsed = []
    for box in bb:
        rectsUsed.append(False)

    acceptedRects = []

    for supIdx, supVal in enumerate(bb):
        if (rectsUsed[supIdx] == False):
            # Initialize current rect
            currxMin = supVal[0]
            currxMax = supVal[2]
            curryMin = supVal[1]
            curryMax = supVal[3]

            # This bounding rect is used
            rectsUsed[supIdx] = True

            # Iterate all initial bounding rects
            # starting from the next
            for subIdx, subVal in enumerate(bb):
                # Initialize merge candidate
                candxMin = subVal[0]
                candxMax = subVal[2]
                candyMin = subVal[1]
                candyMax = subVal[3]

                # Check if x distance between current rect
                # and merge candidate is small enough
                if (doOverlap(currxMin, curryMin, currxMax, curryMax, candxMin, candyMin, candxMax, candyMax)):
                    # Reset coordinates of current rect
                    currxMax = max(candxMax, currxMax)
                    curryMin = min(curryMin, candyMin)
                    curryMax = max(curryMax, candyMax)
                    currxMin = min(currxMin, candxMin)
                    rectsUsed[subIdx] = True
                    # Merge candidate (bounding rect) is used

            # No more merge candidates possible, accept current rect
            acceptedRects.append([currxMin, curryMin, currxMax, curryMax])

    return acceptedRects


# Built-in function for using the pretrained model EAST
model = cv2.dnn.readNet('D:/ocr/frozen_east_text_detection.pb')
pytesseract.pytesseract.tesseract_cmd = r'D:/Tesseract/tesseract.exe'

def process_image(img):

    # The purpose of this step is to make sure that image dimensions are multiples of 32
    # This step is required to make image compatible with EAST algorithm during the
    # branching between PVANet and feature merging

    h, w, c = img.shape
    # print(img.shape)
    # print("Old Dimensions: ", img.shape)

    # print("IMAGE SHAPE: ", img.shape)
    east_h = (h // 32) * 32
    east_w = (w // 32) * 32
    # print("EAST Compatible Dimensions: ", east_h, " ", east_w)

    # Saving the ratio of the dimension change as it will be needed later
    h_ratio = h / east_h
    w_ratio = w / east_w
    # print("Dimension Ratios:", h_ratio, " ", w_ratio)
    # print("Dimension Ratios:", h_ratio, " ", w_ratio)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab", img)

    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)
    # cv2.imshow("l", l)
    # cv2.imshow("a", a)
    # cv2.imshow("b", b)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # cv2.imshow('clahe', cl)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))
    # cv2.imshow('limg', limg)
    # Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # final = cv2.GaussianBlur(final, (5, 5), 0)
    final = cv2.medianBlur(final, 5)
    # cv2.imshow('final', final)

    # blobFromImage(image, scale_factor, size, mean, swapRB, crop)
    blob = cv2.dnn.blobFromImage(final, 1, (east_w, east_h), (123.68, 116.78, 103.94), True, False)

    # From now on image is given to the network
    # Our goal is the extract score and geometry map which will be crucial
    # for loss calculations and removing bounding boxes with low confidence
    model.setInput(blob)

    ln = model.getUnconnectedOutLayersNames()
    # print(ln)
    # ln = ln[::-1]
    # print(ln)

    # To show the layers which the output is needed
    (geometry_map, scores) = model.forward(ln)
    # print(geometry_map.shape)

    # Post processing is done to convert geo maps back to bounding boxes
    # Furthermore thresholding is applied to eliminate boxes with low confidence
    # Finally the non-eliminated boxes are merged
    rectangles = []
    c_scores = []
    scoresData = []
    (numRows, numCols) = scores.shape[2:4]
    # Loop over the number of rows
    for y in range(0, numRows):
        # Extract the scores (probabilities), followed by the geometrical
        scoresData = scores[0, 0, y]
        xData0 = geometry_map[0, 0, y]
        xData1 = geometry_map[0, 1, y]
        xData2 = geometry_map[0, 2, y]
        xData3 = geometry_map[0, 3, y]
        anglesData = geometry_map[0, 4, y]

        # Loop over the number of columns
        for x in range(0, numCols):
            # If the confidence score is under the threshold, do not consider it
            if scoresData[x] < 0.5:
                continue

            # Resulting feature maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # Extract the rotation angle for the prediction
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Derive the width and height of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # Compute both the starting and ending (x, y)-coordinates
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add the bounding box coordinates and probability score to respective lists
            rectangles.append((startX, startY, endX, endY))
            c_scores.append(scoresData[x])

    # print(rectangles)

    # Non_max_suppression
    bb = non_max_suppression(np.array(rectangles), probs=c_scores, overlapThresh=0.9)
    # print(bb)

    final_recs = []
    final_recs = mergeBoxes(bb)
    final_recs = mergeBoxes(final_recs)
    # print(final_recs)
    txt = ""
    # Display images with bb
    img_with_bb = img.copy()
    for (x1, y1, x2, y2) in final_recs:
        # Converting to the old size
        x1 = int(x1 * w_ratio)
        y1 = int(y1 * h_ratio)
        x2 = int(x2 * w_ratio)
        y2 = int(y2 * h_ratio)

        segment = img[y1:y2+4, x1:x2+2, :]
        print("shape", segment.shape)
        if segment.shape[0] == 0 or segment.shape[1] == 0:
            print("I AM HERE")
            continue

        # Make the image compatible
        segment = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)

        # th, segment = cv2.threshold(segment, 128, 192, cv2.THRESH_OTSU)

        th, segment = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # segment = cv2.adaptiveThreshold(segment, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \ cv2.THRESH_BINARY, 11, 2)

        # cv2.imshow("Segment", segment)
        # cv2.waitKey(0)

        # Pass the image to CRNN
        final_t = pytesseract.image_to_string(segment, config=r"--psm 8", lang='eng+tur')
        txt = final_t.strip() + " " + txt + " "
        print(final_t)
        # cv2.rectangle(img_with_bb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(img_with_bb, final_t.strip(), (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)

    # cv2.imshow("Before", img)
    # cv2.imshow("After", img_with_bb)
    # cv2.imshow("After (tf)", img_with_bb2)
    # cv2.waitKey(0)
    print (txt)
    return txt

def _get_image(url):
    img = Image.open(BytesIO(requests.get(url).content))
    return np.array(img)

def _get_image_dir(path):
    return cv2.imread(path)


def convert_jpg(img):
    return img

def is_url_image(url):
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    r = requests.head(url)
    if r.headers["content-type"] in image_formats:
        return True
    return False

"""
THIS PART IS NOT USED. IT IS FOR TRYING THE NON MAX SUPPRESSION FUNCTION OF TENSORFLOW TO CHECK WHETHER THERE WILL BE A
SIGNIFICANT DIFFERENCE WITH THE IMUTILS FUNCTION

selected_indices = tf.image.non_max_suppression(np.array(rectangles), c_scores, 10, iou_threshold=0.5, score_threshold=float('-inf'), name=None)
selected_boxes = tf.gather(np.array(rectangles), selected_indices)
print(selected_boxes)
selected_boxes = selected_boxes.numpy()
img_with_bb2 = img.copy()
for(x1, y1, x2, y2) in selected_boxes:

    # Converting to the old size
    x1 = int(x1 * w_ratio)
    y1 = int(y1 * h_ratio)
    x2 = int(x2 * w_ratio)
    y2 = int(y2 * h_ratio)

    cv2.rectangle(img_with_bb2, (x1, y1), (x2, y2), (0, 0, 255), 2)
"""

