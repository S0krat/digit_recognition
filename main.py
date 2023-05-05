import cv2
import numpy as np
import itertools
from tensorflow import keras
from keras.datasets import mnist


def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 199, (img.shape[0] + img.shape[1]) / 30)
    contours, hierarchy = cv2.findContours(thresh_img,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_contour_in_contours(all_contours):
    for contour in all_contours:
        if contour[2] * contour[3] > 0.5 * img.shape[0] * img.shape[1]:
            all_contours.remove(contour)
    for rec1, rec2 in itertools.permutations(all_contours, 2):
        if rec2[0] >= rec1[0] and rec2[1] >= rec1[1] and rec2[0] < rec1[0] + rec1[2] and rec2[1] < rec1[1] + rec1[3]:
            if rec2 in all_contours:
                all_contours.remove(rec2)
    return all_contours


def extract_letters(image, boxes, out_size=28):
    letters = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for box in boxes:
        (x, y, w, h) = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (125, 125, 0), 2)
        letter_crop = gray[y:y + h, x:x + w]
        size_max = max(w, h)
        letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
        if w > h:
            y_pos = w // 2 - h // 2
            letter_square[y_pos:y_pos + h, 0:w] = letter_crop
        elif w < h:
            x_pos = h // 2 - w // 2
            letter_square[0:h, x_pos:x_pos + w] = letter_crop
        else:
            letter_square = letter_crop

        sized_letter = 255 * np.ones(shape=[out_size, out_size], dtype=np.uint8)
        sized_letter[4:26, 3:25] = cv2.resize(letter_square, (22, 22), interpolation=cv2.INTER_AREA)
        sized_letter = cv2.bitwise_not(sized_letter)
        letters.append((x, y, sized_letter))

    return letters


def mnist_predict_img(model, letter):
    img_arr = np.expand_dims(letter, axis=0)
    # print(img_arr)
    img_arr = img_arr.reshape((1, 28, 28, 1))
    img_arr = img_arr / 255.0

    predict = model.predict(img_arr)
    result = np.argmax(predict)
    return result


img = cv2.imread("img.jpg")

cont = find_contours(img)
bounding_boxes = [cv2.boundingRect(c) for c in cont]

bounding_boxes = detect_contour_in_contours(bounding_boxes)

letters = extract_letters(img, bounding_boxes)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

model = keras.models.load_model('mnist_letters.h5')

# for i in range(200):
#     if mnist_predict_img(model, x_test[i]) == y_test[i]:
#         print(1, end='')
#     else:
#         print(0, end='')

# print(mnist_predict_img(model, letters[0][2]))

for let in letters:
    num = mnist_predict_img(model, let[2])
    cv2.putText(img, str(num), (let[0], let[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 120, 120), 2)

# cv2.imwrite("img_new.jpg", img)
# cv2.imshow('img', cv2.resize(letters[0][2], (280, 280), interpolation=cv2.INTER_AREA))
cv2.imshow('img', img)
cv2.waitKey(0)
