import os

import cv2
import numpy as np
import math
import random

import app
import Preprocess
import PossibleChar

kNearest = cv2.ml.KNearest_create()

# constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

# constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

# other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


def loadKNNDataAndTrainKNN():
    all_contours_with_data = []  # declare empty lists,
    valid_contours_with_data = []  # we will fill these shortly

    try:
        npa_classifications = np.loadtxt("classifications.txt", np.float32)  # read in training classifications
    except:  # if file could not be opened
        print("error, unable to open classifications.txt, exiting program\n")  # show error message
        os.system("pause")
        return False  # and return False

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)  # read in training images
    except:  # if file could not be opened
        print("error, unable to open flattened_images.txt, exiting program\n")  # show error message
        os.system("pause")
        return False  # and return False

    npa_classifications = npa_classifications.reshape(
        (npa_classifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train

    kNearest.setDefaultK(1)  # set default K to 1

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npa_classifications)  # train KNN object

    return True  # if we got here training was successful so return true


def detect_chars_in_plates(list_of_possible_plates):
    int_plate_counter = 0
    img_contours = None
    contours = []

    if len(list_of_possible_plates) == 0:  # if list of possible plates is empty
        return list_of_possible_plates  # return

    # at this point we can be sure the list of possible plates has at least one plate
    for possible_plate in list_of_possible_plates:  # for each possible plate, this is a big for
        # loop that takes up most of the function

        possible_plate.imgGrayscale, possible_plate.imgThresh = Preprocess.preprocess(
            possible_plate.imgPlate)  # preprocess to get grayscale and threshold images

        if app.showSteps:
            cv2.imshow("5a", possible_plate.imgPlate)
            cv2.imshow("5b", possible_plate.imgGrayscale)
            cv2.imshow("5c", possible_plate.imgThresh)

        # increase size of plate image for easier viewing and char detection
        possible_plate.imgThresh = cv2.resize(possible_plate.imgThresh, (0, 0), fx=1.6, fy=1.6)

        # threshold again to eliminate any gray areas
        threshold_value, possible_plate.imgThresh = cv2.threshold(possible_plate.imgThresh, 0.0, 255.0,
                                                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if app.showSteps:
            cv2.imshow("5d", possible_plate.imgThresh)

        # find all possible chars in the plate,
        # this function first finds all contours, then only includes contours that could be chars
        list_of_possible_chars_in_plate = findPossibleCharsInPlate(possible_plate.imgGrayscale,
                                                                   possible_plate.imgThresh)

        if app.showSteps:  # show steps ###################################################
            height, width, num_channels = possible_plate.imgPlate.shape
            img_contours = np.zeros((height, width, 3), np.uint8)
            del contours[:]  # clear the contours list

            for possibleChar in list_of_possible_chars_in_plate:
                contours.append(possibleChar.contour)
            # end for

            cv2.drawContours(img_contours, contours, -1, app.SCALAR_WHITE)

            cv2.imshow("6", img_contours)
        # given a list of all possible chars, find groups of matching chars within the plate
        list_of_lists_of_matching_chars_in_plate = find_list_of_lists_of_matching_chars(list_of_possible_chars_in_plate)

        if app.showSteps:
            img_contours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in list_of_lists_of_matching_chars_in_plate:
                int_random_blue = random.randint(0, 255)
                int_random_green = random.randint(0, 255)
                int_random_red = random.randint(0, 255)

                for matching_char in listOfMatchingChars:
                    contours.append(matching_char.contour)
                cv2.drawContours(img_contours, contours, -1, (int_random_blue, int_random_green, int_random_red))
            cv2.imshow("7", img_contours)

        if len(list_of_lists_of_matching_chars_in_plate) == 0:  # if no groups of matching chars were found in the plate

            if app.showSteps:
                print("chars found in plate number " + str(
                    int_plate_counter) + " = (none), click on any image and press a key to continue . . .")
                int_plate_counter = int_plate_counter + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)

            possible_plate.strChars = ""
            continue  # go back to top of for loop

        for i in range(0, len(list_of_lists_of_matching_chars_in_plate)):  # within each list of matching chars
            list_of_lists_of_matching_chars_in_plate[i].sort(
                key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right
            list_of_lists_of_matching_chars_in_plate[i] = removeInnerOverlappingChars(
                list_of_lists_of_matching_chars_in_plate[i])  # and remove inner overlapping chars

        if app.showSteps:
            img_contours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in list_of_lists_of_matching_chars_in_plate:
                int_random_blue = random.randint(0, 255)
                int_random_green = random.randint(0, 255)
                int_random_red = random.randint(0, 255)

                del contours[:]

                for matching_char in listOfMatchingChars:
                    contours.append(matching_char.contour)
                cv2.drawContours(img_contours, contours, -1, (int_random_blue, int_random_green, int_random_red))
            cv2.imshow("8", img_contours)

        int_len_of_longest_list_of_chars = 0
        int_index_of_longest_list_of_chars = 0

        # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(list_of_lists_of_matching_chars_in_plate)):
            if len(list_of_lists_of_matching_chars_in_plate[i]) > int_len_of_longest_list_of_chars:
                int_len_of_longest_list_of_chars = len(list_of_lists_of_matching_chars_in_plate[i])
                int_index_of_longest_list_of_chars = i

        longest_list_of_matching_chars_in_plate = list_of_lists_of_matching_chars_in_plate[
            int_index_of_longest_list_of_chars]

        if app.showSteps:
            img_contours = np.zeros((height, width, 3), np.uint8)
            del contours[:]
            for matching_char in longest_list_of_matching_chars_in_plate:
                contours.append(matching_char.contour)
            cv2.drawContours(img_contours, contours, -1, app.SCALAR_WHITE)
            cv2.imshow("9", img_contours)
        possible_plate.strChars = recognizeCharsInPlate(possible_plate.imgThresh,
                                                        longest_list_of_matching_chars_in_plate)
        if app.showSteps:
            print("chars found in plate number " + str(
                int_plate_counter) + " = " + possible_plate.strChars +
                  ", click on any image and press a key to continue . . .")
            int_plate_counter = int_plate_counter + 1
            cv2.waitKey(0)

    if app.showSteps:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)
    return list_of_possible_plates


def findPossibleCharsInPlate(img_grayscale, img_thresh):
    list_of_possible_chars = []  # this will be the return value
    contours = []
    img_thresh_copy = img_thresh.copy()

    # find all contours in plate
    contours, npa_hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:  # for each contour
        possible_char = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(
                possible_char):  # if contour is a possible char, note this does not compare to other chars
            list_of_possible_chars.append(possible_char)  # add to list of possible chars
    return list_of_possible_chars


def checkIfPossibleChar(possible_char):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    if (possible_char.intBoundingRectArea > MIN_PIXEL_AREA and
            possible_char.intBoundingRectWidth > MIN_PIXEL_WIDTH and possible_char.intBoundingRectHeight >
            MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possible_char.fltAspectRatio < MAX_ASPECT_RATIO):
        return True
    else:
        return False


def find_list_of_lists_of_matching_chars(list_of_possible_chars):
    # with this function, we start off with all the possible chars in one big list
    # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
    # note that chars that are not found to be in a group of matches do not need to be considered further
    list_of_lists_of_matching_chars = []  # this will be the return value

    for possibleChar in list_of_possible_chars:  # for each possible char in the one big list of chars
        list_of_matching_chars = findListOfMatchingChars(possibleChar,
                                                         list_of_possible_chars)

        list_of_matching_chars.append(possibleChar)  # also add the current char to current
        # possible list of matching chars

        if len(
                list_of_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS:  # if current possible list of matching chars
            # is not long enough to constitute a possible plate
            continue  # jump back to the top of the for loop and try again with next char, note that it's not necessary
            # to save the list in any way since it did not have enough chars to be a possible plate
        # end if

        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        list_of_lists_of_matching_chars.append(list_of_matching_chars)  # so add to our list of lists of matching chars

        list_of_possible_chars_with_current_matches_removed = []

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        list_of_possible_chars_with_current_matches_removed = list(
            set(list_of_possible_chars) - set(list_of_matching_chars))

        recursive_list_of_lists_of_matching_chars = find_list_of_lists_of_matching_chars(
            list_of_possible_chars_with_current_matches_removed)  # recursive call

        for recursiveListOfMatchingChars in recursive_list_of_lists_of_matching_chars:  # for each list of matching chars
            # found by recursive call
            list_of_lists_of_matching_chars.append(
                recursiveListOfMatchingChars)  # add to our original list of lists of matching chars
        break
    return list_of_lists_of_matching_chars


def findListOfMatchingChars(possible_char, list_of_chars):
    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single
    # possible char, and return those matching chars as a list
    list_of_matching_chars = []  # this will be the return value

    for possible_matching_char in list_of_chars:  # for each char in big list
        if possible_matching_char == possible_char:
            continue  # so do not add to list of matches and jump back to top of for loop
        # end if
        # compute stuff to see if chars are a match
        flt_distance_between_chars = distanceBetweenChars(possible_char, possible_matching_char)

        flt_angle_between_chars = angleBetweenChars(possible_char, possible_matching_char)

        flt_change_in_area = float(
            abs(possible_matching_char.intBoundingRectArea - possible_char.intBoundingRectArea)) / float(
            possible_char.intBoundingRectArea)

        flt_change_in_width = float(
            abs(possible_matching_char.intBoundingRectWidth - possible_char.intBoundingRectWidth)) / float(
            possible_char.intBoundingRectWidth)
        flt_change_in_height = float(
            abs(possible_matching_char.intBoundingRectHeight - possible_char.intBoundingRectHeight)) / float(
            possible_char.intBoundingRectHeight)

        # check if chars match
        if (flt_distance_between_chars < (possible_char.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                flt_angle_between_chars < MAX_ANGLE_BETWEEN_CHARS and
                flt_change_in_area < MAX_CHANGE_IN_AREA and
                flt_change_in_width < MAX_CHANGE_IN_WIDTH and
                flt_change_in_height < MAX_CHANGE_IN_HEIGHT):
            list_of_matching_chars.append(
                possible_matching_char)  # if the chars are a match, add the current char to list of matching chars
    return list_of_matching_chars  # return result


# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(firstChar, secondChar):
    int_x = abs(firstChar.intCenterX - secondChar.intCenterX)
    int_y = abs(firstChar.intCenterY - secondChar.intCenterY)

    return math.sqrt((int_x ** 2) + (int_y ** 2))


# use basic trigonometry (SOH CAH TOA) to calculate angle between chars


def angleBetweenChars(firstChar, secondChar):
    flt_adj = float(abs(firstChar.intCenterX - secondChar.intCenterX))
    flt_opp = float(abs(firstChar.intCenterY - secondChar.intCenterY))

    if flt_adj != 0.0:  # check to make sure we do not divide by zero if the
        # center X positions are equal, float division by zero will cause a crash in Python
        flt_angle_in_rad = math.atan(flt_opp / flt_adj)  # if adjacent is not zero, calculate angle
    else:
        flt_angle_in_rad = 1.5708
    # end if

    flt_angle_in_deg = flt_angle_in_rad * (180.0 / math.pi)  # calculate angle in degrees

    return flt_angle_in_deg


# if we have two chars overlapping or to close to each other
# to possibly be separate chars, remove the inner (smaller) char,
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be
# found as contours, but we should only include the char once

def removeInnerOverlappingChars(list_of_matching_chars):
    list_of_matching_chars_with_inner_char_removed = list(list_of_matching_chars)  # this will be the return value

    for currentChar in list_of_matching_chars:
        for otherChar in list_of_matching_chars:
            if currentChar != otherChar:  # if current char and other char are not the same char . . .
                # if current char and other char have center points at almost the same location . . .
                if distanceBetweenChars(currentChar, otherChar) < (
                        currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    # if we get in here we have found overlapping chars
                    # next we identify which char is smaller, then if that char was not
                    # already removed on a previous pass, remove it
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:  # if current c
                        # har is smaller than other char
                        if currentChar in list_of_matching_chars_with_inner_char_removed:  # if current char was
                            # not already removed on a previous pass . . .
                            list_of_matching_chars_with_inner_char_removed.remove(
                                currentChar)  # then remove current char
                        # end if
                    else:  # else if other char is smaller than current char
                        if otherChar in list_of_matching_chars_with_inner_char_removed:  # if other char was not
                            # already removed on a previous pass . . .
                            list_of_matching_chars_with_inner_char_removed.remove(otherChar)  # then remove other char

    return list_of_matching_chars_with_inner_char_removed


# this is where we apply the actual char recognition
def recognizeCharsInPlate(img_thresh, list_of_matching_chars):
    str_chars = ""  # this will be the return value, the chars in the lic plate

    height, width = img_thresh.shape

    img_thresh_color = np.zeros((height, width, 3), np.uint8)

    list_of_matching_chars.sort(key=lambda matching_char: matching_char.intCenterX)  # sort chars from left to right

    cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR,
                 img_thresh_color)  # make color version of threshold image so we can draw contours in color on it

    for currentChar in list_of_matching_chars:  # for each char in plate
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth),
               (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(img_thresh_color, pt1, pt2, app.SCALAR_GREEN, 2)  # draw green box around the char

        # crop char out of threshold image
        img_roi = img_thresh[
                  currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                  currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        img_roi_resized = cv2.resize(img_roi, (
            RESIZED_CHAR_IMAGE_WIDTH,
            RESIZED_CHAR_IMAGE_HEIGHT))  # resize image, this is necessary for char recognition

        npa_roi_resized = img_roi_resized.reshape(
            (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))  # flatten image into 1d numpy array

        npa_roi_resized = np.float32(npa_roi_resized)  # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npa_results, neigh_resp, dists = kNearest.findNearest(npa_roi_resized, k=1)

        str_current_char = str(chr(int(npa_results[0][0])))  # get character from results

        str_chars = str_chars + str_current_char  # append current char to full string

    if app.showSteps:
        cv2.imshow("10", img_thresh_color)
    return str_chars
