import numpy as np
import cv2 as cv
import math

rostro = cv.CascadeClassifier('haarcascade_smile.xml')
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in rostros:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        frame2 = frame[y:y + h, x:x + w]
        frame2 = cv.resize(frame2, (80, 80), interpolation=cv.INTER_AREA)
        gray_face = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        cv.imshow('Rostro', frame2)
        cv.imshow('Gray', gray_face)

        total_pixels_face = w * h
        print(f'Tamaño del rostro: {total_pixels_face} píxeles.')

        min_gray = int(0.5 * np.mean(gray_face))
        max_gray = int(1.5 * np.mean(gray_face))
        mask = cv.inRange(gray_face, min_gray, max_gray)
        count_pixels_in_range = cv.countNonZero(mask)
        cv.imshow('Mask', mask)
        print(f'Píxeles dentro del rango de grises [{min_gray}, {max_gray}]: {count_pixels_in_range}')

    cv.imshow('Rostros', frame)
    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()
