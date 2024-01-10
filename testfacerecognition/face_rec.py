import cv2
import face_recognition_models

#chon hinh
img = cv2.imread('./image/Lionel-Messi-2.jpg')
#
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2GB)
#co the load duoc nhieu hinh nhung chung ta test thu 1 hinh truoc

img_encoding = face_recognition_models.face_encodings(rgb_img)[0]

cv2.imshow("Img", img)
cv2.waitKey(0)