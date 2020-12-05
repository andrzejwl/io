import cv2
import numpy as np
import math
import Main
import random

import Preprocess
import DetectChars
import PossiblePlate
import PossibleChar

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def detectPlatesInScene(img_original_scene):
    list_of_possible_plates = []
    height, width, num_channels = img_original_scene.shape

    img_grayscale_scene = np.zeros((height, width, 1), np.uint8)
    img_thresh_scene = np.zeros((height, width, 1), np.uint8)
    img_contours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Main.showSteps:  # show steps
        cv2.imshow("0", img_original_scene)

    img_grayscale_scene, img_thresh_scene = Preprocess.preprocess(
        img_original_scene)  # preprocess to get grayscale and threshold images

    if Main.showSteps:  # show steps
        cv2.imshow("1a", img_grayscale_scene)
        cv2.imshow("1b", img_thresh_scene)

    # find all possible chars in the scene,
    # this function first finds all contours, then only includes
    # contours that could be chars (without comparison to other chars yet)
    list_of_possible_chars_in_scene = findPossibleCharsInScene(img_thresh_scene)

    if Main.showSteps:  # show steps
        print("step 2 - len(list_of_possible_chars_in_scene) = " + str(
            len(list_of_possible_chars_in_scene)))  # 131 with MCLRNF1 image

        img_contours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in list_of_possible_chars_in_scene:
            contours.append(possibleChar.contour)

        cv2.drawContours(img_contours, contours, -1, Main.SCALAR_WHITE)
        cv2.imshow("2b", img_contours)

    # given a list of all possible chars, find groups of matching chars
    # in the next steps each group of matching chars will attempt to be recognized as a plate
    list_of_lists_of_matching_chars_in_scene = DetectChars.findListOfListsOfMatchingChars(
        list_of_possible_chars_in_scene)

    if Main.showSteps:  # show steps
        print("step 3 - list_of_lists_of_matching_chars_in_scene.Count = " + str(
            len(list_of_lists_of_matching_chars_in_scene)))

        img_contours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in list_of_lists_of_matching_chars_in_scene:
            int_random_blue = random.randint(0, 255)
            int_random_green = random.randint(0, 255)
            int_random_red = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)

            cv2.drawContours(img_contours, contours, -1, (int_random_blue, int_random_green, int_random_red))

        cv2.imshow("3", img_contours)

    for listOfMatchingChars in list_of_lists_of_matching_chars_in_scene:  # for each group of matching chars
        possible_plate = extractPlate(img_original_scene, listOfMatchingChars)  # attempt to extract plate

        if possible_plate.imgPlate is not None:  # if plate was found
            list_of_possible_plates.append(possible_plate)  # add to list of possible plates

    print("\n" + str(len(list_of_possible_plates)) + " possible plates found")

    if Main.showSteps:
        print("\n")
        cv2.imshow("4a", img_contours)

        for i in range(0, len(list_of_possible_plates)):
            p2f_rect_points = cv2.boxPoints(list_of_possible_plates[i].rrLocationOfPlateInScene)

            cv2.line(img_contours, tuple(p2f_rect_points[0]), tuple(p2f_rect_points[1]), Main.SCALAR_RED, 2)
            cv2.line(img_contours, tuple(p2f_rect_points[1]), tuple(p2f_rect_points[2]), Main.SCALAR_RED, 2)
            cv2.line(img_contours, tuple(p2f_rect_points[2]), tuple(p2f_rect_points[3]), Main.SCALAR_RED, 2)
            cv2.line(img_contours, tuple(p2f_rect_points[3]), tuple(p2f_rect_points[0]), Main.SCALAR_RED, 2)

            cv2.imshow("4a", img_contours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", list_of_possible_plates[i].imgPlate)
            cv2.waitKey(0)

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)

    return list_of_possible_plates


def findPossibleCharsInScene(imgThresh):
    list_of_possible_chars = []  # this will be the return value

    int_count_of_possible_chars = 0

    img_thresh_copy = imgThresh.copy()

    contours, npa_hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)  # find all contours

    height, width = imgThresh.shape
    img_contours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):  # for each contour

        if Main.showSteps:
            cv2.drawContours(img_contours, contours, i, Main.SCALAR_WHITE)

        possible_char = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(
                possible_char):  # if contour is a possible char, note this does not compare to other chars
            int_count_of_possible_chars = int_count_of_possible_chars + 1  # increment count of possible chars
            list_of_possible_chars.append(possible_char)  # and add to list of possible chars

    if Main.showSteps:
        print("\nstep 2 - len(contours) = " + str(len(contours)))
        print("step 2 - int_count_of_possible_chars = " + str(int_count_of_possible_chars))
        cv2.imshow("2a", img_contours)

    return list_of_possible_chars


def extractPlate(img_original, list_of_matching_chars):
    possible_plate = PossiblePlate.PossiblePlate()  # this will be the return value

    list_of_matching_chars.sort(
        key=lambda matching_char: matching_char.intCenterX)  # sort chars from left to right based on x position

    # calculate the center point of the plate
    flt_plate_center_x = (list_of_matching_chars[0].intCenterX + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intCenterX) / 2.0
    flt_plate_center_y = (list_of_matching_chars[0].intCenterY + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intCenterY) / 2.0

    pt_plate_center = flt_plate_center_x, flt_plate_center_y

    # calculate plate width and height
    int_plate_width = int(
        (list_of_matching_chars[len(list_of_matching_chars) - 1].intBoundingRectX + list_of_matching_chars[
            len(list_of_matching_chars) - 1].intBoundingRectWidth - list_of_matching_chars[
             0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    int_total_of_char_heights = 0

    for matchingChar in list_of_matching_chars:
        int_total_of_char_heights = int_total_of_char_heights + matchingChar.intBoundingRectHeight
    # end for

    flt_average_char_height = int_total_of_char_heights / len(list_of_matching_chars)

    int_plate_height = int(flt_average_char_height * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    flt_opposite = list_of_matching_chars[len(list_of_matching_chars) - 1].intCenterY - list_of_matching_chars[
        0].intCenterY
    flt_hypotenuse = DetectChars.distanceBetweenChars(list_of_matching_chars[0],
                                                      list_of_matching_chars[len(list_of_matching_chars) - 1])
    flt_correction_angle_in_rad = math.asin(flt_opposite / flt_hypotenuse)
    flt_correction_angle_in_deg = flt_correction_angle_in_rad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possible_plate.rrLocationOfPlateInScene = (
        tuple(pt_plate_center), (int_plate_width, int_plate_height), flt_correction_angle_in_deg)

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotation_matrix = cv2.getRotationMatrix2D(tuple(pt_plate_center), flt_correction_angle_in_deg, 1.0)

    height, width, num_channels = img_original.shape  # unpack original image width and height

    img_rotated = cv2.warpAffine(img_original, rotation_matrix, (width, height))  # rotate the entire image

    img_cropped = cv2.getRectSubPix(img_rotated, (int_plate_width, int_plate_height), tuple(pt_plate_center))

    possible_plate.imgPlate = img_cropped  # copy the cropped plate image
    # into the applicable member variable of the possible plate

    return possible_plate
