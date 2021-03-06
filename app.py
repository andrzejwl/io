import cv2
import os
import io
import math
from datetime import datetime

import detect_chars
import detect_plates
import pytesseract

# read tesseract executable location
tesseract_file = open('tesseract.txt', 'r')
tesseract_dir = tesseract_file.readline()

pytesseract.pytesseract.tesseract_cmd = r'{}'.format(tesseract_dir)

SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


def main(input_file_path):
    bln_knn_training_successful = detect_chars.loadKNNDataAndTrainKNN()  # attempt KNN training

    if not bln_knn_training_successful:
        print("\nerror: KNN traning was not successful\n")
        return

    # img_original_scene  = cv2.imread("resources/1.png")
    vidcap = cv2.VideoCapture(input_file_path)
    output_video = cv2.VideoWriter('PV/output' + datetime.now().strftime('%D %H:%M:%S') + '.mp4', -1, 30.0, (1280,720))
    success, img_original_scene = vidcap.read()

    detected_plates_txt = []

    second_of_recording = 0
    frame = 0
    counter = 0

    plate_to_draw = None  # last detected plate

    while success:
        frame = frame + 1
        if counter % 10 == 0:
            second_of_recording = math.floor(frame / 270)
            if img_original_scene is None:
                print("\nerror: image not read from file \n\n")
                os.system("pause")
                return

            list_of_possible_plates = detect_plates.detectPlatesInScene(img_original_scene)  # detect plates

            plates_to_check = list_of_possible_plates[:3]

            possible_plates = []
            for p in plates_to_check:
                text = pytesseract.image_to_string(p.imgPlate)
                if 5 < len(text) < 8:
                    possible_plates.append("\"" + ''.join(ch for ch in text if ch.isalnum()) + "\" in second: " + str(second_of_recording))

            if len(possible_plates):
                for plate in possible_plates:
                    detected_plates_txt.append(plate)
                print(possible_plates)
                # draw the most probable frame

                plate_to_draw = plates_to_check[0]
                out_frame = drawRedRectangleAroundPlate(img_original_scene.copy(), plate_to_draw)
                output_video.write(out_frame)
            else:
                plate_to_draw = None
                output_video.write(img_original_scene)

            success, img_original_scene = vidcap.read()
        else:
            output_frame = img_original_scene.copy()
            if plate_to_draw:
                drawRedRectangleAroundPlate(output_frame, plate_to_draw)
            output_video.write(output_frame)
        counter = counter + 1

    try:
        f = io.open("license_plates.txt", "x", encoding="utf-8")
    except:
        print("File exist")

    f = io.open("license_plates.txt", "w", encoding="utf-8")
    for plate in detected_plates_txt:
        f.write(plate)
        f.write("\n")
    f.close()
    return


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2f_rect_points = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect

    # draw 4 red lines
    cv2.line(imgOriginalScene, tuple(p2f_rect_points[0]), tuple(p2f_rect_points[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2f_rect_points[1]), tuple(p2f_rect_points[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2f_rect_points[2]), tuple(p2f_rect_points[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2f_rect_points[3]), tuple(p2f_rect_points[0]), SCALAR_RED, 2)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    pt_center_of_text_area_x = 0  # this will be the center of the area the text will be written to
    pt_center_of_text_area_y = 0

    pt_lower_left_text_origin_x = 0  # this will be the bottom left of the area that the text will be written to
    pt_lower_left_text_origin_y = 0

    scene_height, scene_width, scene_num_channels = imgOriginalScene.shape
    plate_height, plate_width, plate_num_channels = licPlate.imgPlate.shape

    int_font_face = cv2.FONT_HERSHEY_SIMPLEX  # choose a plain jane font
    flt_font_scale = float(plate_height) / 30.0  # base font scale on height of plate area
    int_font_thickness = int(round(flt_font_scale * 1.5))  # base font thickness on font scale

    text_size, baseline = cv2.getTextSize(licPlate.strChars, int_font_face, flt_font_scale, int_font_thickness)

    # unpack roatated rect into center point, width and height, and angle
    ((int_plate_center_x, int_plate_center_y), (int_plate_width, int_plate_height),
     flt_correction_angle_in_deg) = licPlate.rrLocationOfPlateInScene

    int_plate_center_x = int(int_plate_center_x)  # make sure center is an integer
    int_plate_center_y = int(int_plate_center_y)

    pt_center_of_text_area_x = int(
        int_plate_center_x)  # the horizontal location of the text area is the same as the plate

    if int_plate_center_y < (scene_height * 0.75):  # if the license plate is in the upper 3/4 of the image
        pt_center_of_text_area_y = int(round(int_plate_center_y)) + int(
            round(plate_height * 1.6))  # write the chars in below the plate
    else:
        pt_center_of_text_area_y = int(round(int_plate_center_y)) - int(
            round(plate_height * 1.6))  # write the chars in above the plate

    text_size_width, text_size_height = text_size

    pt_lower_left_text_origin_x = int(
        pt_center_of_text_area_x - (text_size_width / 2))  # calculate the lower left origin of the text area
    pt_lower_left_text_origin_y = int(
        pt_center_of_text_area_y + (text_size_height / 2))  # based on the text area center, width, and height

    cv2.putText(imgOriginalScene, licPlate.strChars, (pt_lower_left_text_origin_x, pt_lower_left_text_origin_y),
                int_font_face, flt_font_scale, SCALAR_YELLOW, int_font_thickness)


if __name__ == "__main__":
    main('resources/grupaA3.mp4')
