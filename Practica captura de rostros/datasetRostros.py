import numpy as np
import cv2 as cv
import math

rostro = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv.VideoCapture(0)
i = 0

while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rostros = rostro.detectMultiScale(gray, 1.3, 5)


    for (x, y, w, h) in rostros:
         #cv.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
        frame2 = frame[y:y+h, x:x+w]
        # frame3 = frame[x+30:x+w-30, y+30:y+h-30]
        frame2 = cv.resize(frame2, (100, 100), interpolation=cv.INTER_AREA)
        cv.imwrite('output/rostro' + str(i) + '.jpg', frame2)
        cv.imshow('rostror', frame2)

    
    cv.imshow('rostros', frame)
    i += 1
    k = cv.waitKey(1)
    if k == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv.destroyAllWindows()


# actividad 1 generar el recorte de cualquier objeto
# recorte de caras pasarlas a binario y contarlos para ver si la cantidad de blancos aumenta o decrementa si haces un gesto
