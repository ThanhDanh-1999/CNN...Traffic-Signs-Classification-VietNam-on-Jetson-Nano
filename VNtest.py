import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

#############################################

frameWidth = 640  # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# IMPORT THE TRANNIED MODEL
model = tf.keras.models.load_model('VNmodel.h5')


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


def area_check(contours):
    cnts = list()
    index = list()
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) >= 1024:
            cnts.append(contours[i])
            index.append(i)
    return cnts, index


def shape_check(contours, index):
    cnts = list()
    index = list()
    for cnt in contours:
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3 or 9 <= len(approx) <= 12:
            cnts.append(cnt)
            index.append(contours.index(cnt))
    return cnts, index


def draw_rect(contours, image):
    for cnt in contours:
        x, y, w, h, = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


# DETECT TRAFFIC SIGN
def detect_traffic_sign(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Detect red upper hue values (TODO immprovement)
    lower_red2 = np.array([160, 0, 00])
    upper_red2 = np.array([179, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    res1 = cv2.bitwise_and(image, image, mask=mask1)

    # Detect lower red hue values
    lower_red = np.array([0, 100, 150])
    upper_red = np.array([10, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red, upper_red)
    res2 = cv2.bitwise_and(image, image, mask=mask2)

    # Combine the above two thresholds to get masked image
    mask_image = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0)

    # Detect contours for mask image
    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area filter for the contours
    area_contours, index = area_check(contours)

    # Shape filtering contours
    shape_contours, index = shape_check(area_contours, index)

    # Hierarchy filters for contours
    # index = hierarchy_check(hierarchy, index)

    # Draw bounding rectangle for contours
    image = draw_rect(shape_contours, image)

    # Crop detected sign and resize to 32x32 pixels
    images = []
    for cnt in shape_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        images.append(cv2.resize(image[y:y + h, x:x + h], (32, 32), interpolation=cv2.INTER_AREA))
    return images


def getCalssName(classNo):
    if classNo == 0:
        return 'Toc do toi da cho phep 20km/h'
    elif classNo == 1:
        return 'Toc do toi da cho phep 40km/h'
    elif classNo == 2:
        return 'Toc do toi da cho phep 50km/h'
    elif classNo == 3:
        return 'Toc do toi da cho phep 60km/h'
    elif classNo == 4:
        return 'Cam di nguoc chieu'
    elif classNo == 5:
        return 'Cam re trai va xe quay dau'
    elif classNo == 6:
        return 'Cam re phai'
    elif classNo == 7:
        return 'Cam quay xe va khong cam re trai'
    elif classNo == 8:
        return 'Cam dung xe va do xe'
    elif classNo == 9:
        return 'Duong nguoi di bo cat ngang'
    elif classNo == 10:
        return 'Cam oto tai vuot'
    elif classNo == 11:
        return 'Duong giao nhau cung cap'
    elif classNo == 12:
        return 'Giao nhau voi duong khong uu tien'
    elif classNo == 13:
        return 'Giao nhau voi duong uu tien'
    elif classNo == 14:
        return 'Dung lai'
    elif classNo == 15:
        return 'Cong truong'
    elif classNo == 16:
        return 'Di cham'
    elif classNo == 17:
        return 'Huong phai di vong chuong ngai vat'
    elif classNo == 18:
        return 'Duong danh cho nguoi di bo'
    elif classNo == 19:
        return 'Duong mot chieu'
    elif classNo == 20:
        return 'Tram cung cap xang dau'


def regcognize():
    if probabilityValue > threshold:
        print(getCalssName(classIndex))
        cv2.putText(imgOrignal, "CLASS: ", (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35), font, 0.75,
                    (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)


while True:
    success, imgOrignal = cap.read()
    #success, imgOrignal = cv2.imread('124.png', cv2.IMREAD_COLOR)
    images = detect_traffic_sign(imgOrignal)
    if len(images) != 0:
        for image in images:
            # img = np.asarray(imgOrignal)
            # img = cv2.resize(imgOrignal, (32, 32))
            img = preprocessing(image)
            cv2.imshow("1", cv2.resize(img, (100, 100)))
            img = img.reshape(1, 32, 32, 1)
            predictions = model.predict(img)
            classIndex = model.predict_classes(img)
            probabilityValue = np.amax(predictions)
            cv2.imshow("Result", imgOrignal)
            regcognize()
    cv2.imshow("Result", imgOrignal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()