import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('wulitaidou.jpg')
faces = face_cascade.detectMultiScale(image = img, scaleFactor = 1.1, minNeighbors = 5)
for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
print(len(faces),"faces detected!")
finalimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

plt.figure(figsize=(12,12))
plt.imshow(finalimg) 
plt.axis("off")
plt.show()