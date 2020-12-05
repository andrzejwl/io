import cv2
import imutils
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
vidcap = cv2.VideoCapture('grupaA1.mp4')

success, image = vidcap.read()
count = 0

while success:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.GaussianBlur(image_gray, (5,5), 0)

    edges = cv2.Canny(image_gray, 30, 200)

    contours, new = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if count == 120:
        print(len(contours))
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]
    #contours = imutils.grab_contours(contours)

    if count == 100:
        img2 = image_gray.copy()
        threshold = np.array([30,30,30])
        white = np.array([200,200,200])
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, threshold, white)

        result = cv2.bitwise_and(image, image, mask=mask)
        result[mask==0] = (255,255,255)

        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
        cv2.imshow('mask applied', result)
        cv2.drawContours(img2, contours, -1 ,(0,255,0), 3)
        cv2.imshow('biggest contours', img2)
        cv2.waitKey(0)
        #break

    contour_with_license_plate = None
    license_plate = None
    x = None
    y = None
    w = None
    h = None

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4:
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            license_plate = image_gray[y:y + h, x:x + w]
            break

    if license_plate is None:
        count += 1
        success, image = vidcap.read()
        continue

    license_plate = cv2.bilateralFilter(license_plate, 11, 17, 17)
    (thresh, license_plate) = cv2.threshold(license_plate, 150, 180, cv2.THRESH_BINARY)

    text = pytesseract.image_to_string(license_plate)

    if len(text) > 1:
        print(text)

    count += 1
    success, image = vidcap.read()

