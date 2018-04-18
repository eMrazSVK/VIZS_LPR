import cv2
import numpy as np
import pytesseract


# return bounding boxes of plate candidates and [optional -show processing]
def detect_plate_candidates(input_image, show_processing):
    width, height, channels = input_image.shape

    # init help arrays (images)
    black = np.zeros((width, height, 1), dtype="uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # very basic image pre-processing
    blur = cv2.medianBlur(gray, 3)

    # calculate edges
    edges = cv2.Canny(blur, 100, 120)

    # init of help arrays, variables (arrays)
    contours_area = []
    useful_contours = []
    useful_boxes = []
    row = 0

    # contours - just convert edges to contours, so we can calculate area etc
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # iterate through contours (edges) of picture, rules for accepting contour as a plate candidate:
    # a) contour area cannot be small (reflections, small hotspots etc) or large (car hood reflection, white car)
    # b) Width and Height ratio of contour should fit Width/Height ratio of Slovak license plate (approximately)
    # c) contour have to be Convex (enclosed)
    for contour in contours:
        if (cv2.contourArea(contour) < 300) or (cv2.contourArea(contour) > 8000) and not cv2.isContourConvex(contour):
            contours_area.append(0)
        else:
            x, y, w, h = cv2.boundingRect(contour)
            if (w / h > 4) and (w / h < 6):
                contours_area.append(cv2.contourArea(contour))
                useful_contours.append(contour)
                useful_boxes.append([])
                useful_boxes[row].append([x, y, w, h])
                row += 1

    print(useful_boxes)
    # crop plate candidate
    if len(useful_boxes) is not 0:
        license_plate = img[useful_boxes[0][0][1]:(useful_boxes[0][0][1] + useful_boxes[0][0][3]),
                            useful_boxes[0][0][0]:(useful_boxes[0][0][0] + useful_boxes[0][0][2])]

        if show_processing:
            cv2.drawContours(black, useful_contours, -1, 255, 1)
            cv2.imshow('Plate candidates (contours)', black)
            cv2.imshow('Raw Image', img)
            cv2.imshow('Best Plate Candidate', license_plate)
            cv2.waitKey(0)

        return license_plate

    else:
        return img


# do OCR within ROI's in input image (ROI's gathered from detect_plate_candidates()), return characters recognized
# also do some processing within ROI (treshold)
def plate_candidates_ocr(plate_candidates):
    # apply mean-shift-filter, slow, but helps alot
    plate_candidates = cv2.pyrMeanShiftFiltering(plate_candidates,60.0,60.0)
    plate_candidates = cv2.cvtColor(plate_candidates, cv2.COLOR_RGB2GRAY)
    plate_candidates = cv2.threshold(plate_candidates, 80, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)[1]
    cv2.imshow('Best Plate Candidate', plate_candidates)
    text = pytesseract.image_to_string(plate_candidates)
    return text

# read input image, change input to fit your needs
img = cv2.imread('test1.jpg', 1)
x = detect_plate_candidates(img, True)
recognized_text = plate_candidates_ocr((x))
print(recognized_text)
cv2.waitKey(0)
