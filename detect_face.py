import cv2
import numpy as np

def detect_face(img):
  face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
  face_img = img.copy()
  face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)

  for i, (x,y,w,h) in enumerate(face_rects):
    rect_face_img = face_img[x:w, y:h]
    file_name = 'face'+ i.zfill(8) + '.png'
    cv2.imwrite(file_name, rect_face_img)
  return
  