import cv2
import numpy as np
import pytesseract
import threading
import os


# global buffer used for stacking frames with possibility of having license plate in them
lp_image_buffer = []

# global counters for writing/reading to/from lp_image_buffer
write_count = 0
read_count  = 0

# path to images and final detected plates
IMAGE_PATH = '/home/eduard/PycharmProjects/hrdcode_lpr/lp_candidates/'
FINAL_PATH = '/home/eduard/PycharmProjects/hrdcode_lpr/detected/'

# define font properties for text to be written in images
font      = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255,0,0)
lineType  = 2


# return bounding boxes of plate candidates and [optional -show processing]
def detect_plate_candidates(input_image, show_processing, num):
    global write_count
    width, height, channels = input_image.shape

    # init help arrays (images)
    black = np.zeros((width, height, 1), dtype="uint8")
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

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
    im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # iterate through contours (edges) of picture, rules for accepting contour as a plate candidate:
    # a) contour area cannot be small (reflections, small hot spots etc) or large (car hood reflection, white car)
    # b) Width and Height ratio of contour should fit Width/Height ratio of Slovak license plate (approximately)
    # c) contour have to be Convex (enclosed)
    # d) ratio of black/white pixels in threshold image have to match b/w ratio of license plate
    for contour in contours:
        # we assume that plate is rectangular shape, so its bounding box is almost exact shape of the plate
        # so area of contour's bounding box and area of contour itself should not vary too much
        x, y, w, h = cv2.boundingRect(contour)
        bounding_box_area = w*h
        contour_area = cv2.contourArea(contour)
        area_err_tolerance = 0.1*contour_area

        # or (cv2.contourArea(contour) > 8000): # and not cv2.isContourConvex(contour):
        if contour_area < 300 \
           and not cv2.isContourConvex(contour) \
           and not abs(bounding_box_area - contour_area) < area_err_tolerance:
            pass

        else:
            if (w / h > 4) and (w / h < 6):
                useful_contours.append(contour)
                useful_boxes.append([])
                useful_boxes[row].append([x, y, w, h])
                row += 1

    # crop plate candidate
    if len(useful_boxes) is not 0:
        license_plate = input_image[useful_boxes[0][0][1]:(useful_boxes[0][0][1] + useful_boxes[0][0][3]),
                                    useful_boxes[0][0][0]:(useful_boxes[0][0][0] + useful_boxes[0][0][2])]

        #  convert to gray and threshold, so we can calculate number of white pixels
        thresh_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        thresh_plate = cv2.threshold(thresh_plate, 110, 255, cv2.THRESH_BINARY)[1]

        # calculate actual ROI area
        plate_area = useful_boxes[0][0][3] * useful_boxes[0][0][2]
        white_pixel_count = np.sum(thresh_plate == 255)

        # black/white ratio have to match black/white ratio of license plate
        if 0.60 < white_pixel_count/plate_area < 0.80:

            if show_processing:
                cv2.drawContours(black, useful_contours, -1, 255, 1)
                cv2.imshow('Plate candidates (contours)', thresh_plate)
                cv2.imshow('Best Plate Candidate', license_plate)
                cv2.imshow('Raw Image', input_image)

            cv2.imwrite(os.path.join(IMAGE_PATH, str(write_count)+'.jpg'), license_plate)
            cv2.imwrite(os.path.join(FINAL_PATH, str(write_count) + '.jpg'), input_image)
            write_count += 1
            return 0

        else:
            return -1


# do OCR within ROI's in input image (ROI's gathered from detect_plate_candidates()), return characters recognized
# also do some processing within ROI (threshold)
def plate_candidates_ocr():
    global read_count
    # apply mean-shift-filter, slow, but helps a lot
    # plate_candidates = cv2.pyrMeanShiftFiltering(plate_candidates,60.0,60.0)

    samples = np.loadtxt('char_samples.data', np.float32)
    responses = np.loadtxt('char_responses.data', np.float32)
    responses = responses.reshape((responses.size, 1))

    model = cv2.ml.KNearest_create()
    model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    while True:
        print(str(read_count))
        if read_count < write_count:
            for i in range(0, write_count-read_count):
                plate_candidate = cv2.imread(IMAGE_PATH+str(read_count)+'.jpg', 1)
                img = clean_image(plate_candidate)
                clean_img, chars = extract_characters(img)
                plate_chars = ""
                read_count += 1

                for bbox, char_img in chars:
                    width, height = char_img.shape
                    if width is not 0 and height is not 0:
                        small_img = cv2.resize(char_img, (10, 10))
                        small_img = small_img.reshape((1, 100))
                        small_img = np.float32(small_img)
                        retval, results, neigh_resp, dists = model.findNearest(small_img, k=1)
                        plate_chars += str(chr((results[0][0])))
                
                if plate_chars is not "":
                    original_image = cv2.imread(FINAL_PATH + str(read_count) + '.jpg', 1)
                    cv2.putText(original_image, plate_chars,(50,50), font, fontScale, fontColor, lineType)
                    cv2.imwrite(FINAL_PATH + str(read_count) + '.jpg', original_image)
                else:
                    pass


def reduce_colors(img, n):
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2


def clean_image(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img
        , None
        , fx=5.0
        , fy=5.0
        , interpolation=cv2.INTER_CUBIC)

    resized_img = cv2.GaussianBlur(resized_img,(5,5),0)
    cv2.imwrite('licence_plate_large.png', resized_img)

    equalized_img = cv2.equalizeHist(resized_img)
    cv2.imwrite('licence_plate_equ.png', equalized_img)


    reduced = cv2.cvtColor(reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8), cv2.COLOR_BGR2GRAY)
    cv2.imwrite('licence_plate_red.png', reduced)


    ret, mask = cv2.threshold(reduced, 80, 255, cv2.THRESH_BINARY)
    cv2.imwrite('licence_plate_mask.png', mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations = 1)
    cv2.imwrite('licence_plate_mask2.png', mask)

    return mask


def extract_characters(img):
    bw_image = cv2.bitwise_not(img)
    contours = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]

    char_mask = np.zeros_like(img)
    bounding_boxes = []
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        area = w * h
        center = (x + w/2, y + h/2)
        if (area > 1000) and (area < 10000) and (h > 60):
            x,y,w,h = x-4, y-4, w+8, h+8
            bounding_boxes.append((center, (x,y,w,h)))
            cv2.rectangle(char_mask,(x,y),(x+w,y+h),255,-1)

    cv2.imwrite('licence_plate_mask3.png', char_mask)

    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask = bw_image))

    bounding_boxes = sorted(bounding_boxes, key=lambda item: item[0][0])

    characters = []
    for center, bbox in bounding_boxes:
        x,y,w,h = bbox
        char_image = clean[y:y+h,x:x+w]
        characters.append((bbox, char_image))

    return clean, characters


def highlight_characters(img, chars):
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for bbox, char_img in chars:
        x,y,w,h = bbox
        cv2.rectangle(output_img,(x,y),(x+w,y+h),255,1)

    return output_img


# main function
def main():
    cap = cv2.VideoCapture('testVid3.mp4')
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        width, height, channels = frame.shape
        frame = cv2.resize(frame, (int(0.4 * height), int(0.4 * width)))


        if counter % 1 == 0:
            detect_plate_candidates(frame, True, counter)

        cv2.imshow('Actual Frame', frame)
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break


# Create 2 threads for 'pseudo RT' functionality:
# a) Process captured framer from camera and save them to buffer
# b) Slowly do OCR on plate candidates from thread a), save the results to folder together with corresponding pictures
thread_a = threading.Thread(target=main)
thread_b = threading.Thread(target=plate_candidates_ocr)

thread_a.daemon = True
thread_b.daemon = True

thread_a.start()
thread_b.start()

thread_a.join()
thread_b.join()

# working with single image
'''
# read input image, change input to fit your needs
img = cv2.imread('IMG_0549.jpg', 1)
# main functionality
x = detect_plate_candidates(img, True, 1)

if x is not -1:
    recognized_text = plate_candidates_ocr((x))
    print(recognized_text)

cv2.waitKey(0)
'''

