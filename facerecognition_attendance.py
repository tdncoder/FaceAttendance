import cv2
import numpy as np
import face_recognition

#sửa file hình import ở đây
imgElon = face_recognition.load_image_file("./image/ElonMusk.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("./image/ElonTest.jpg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#Địa chỉ nhận dạng mặt
faceLocation = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
# kẻ khung nhận dạng mặt xác định tọa độ XY để kẻ khung sao cho vừa với khuôn mặt
cv2.rectangle(imgElon, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255,0,0), 2 )
# Địa chỉ nhận dạng mặt
faceLocation = face_recognition.face_locations(imgTest)[0]
encodeElonTest = face_recognition.face_encodings(imgTest)[0]
# kẻ khung nhận dạng mặt xác định tọa độ XY để kẻ khung sao cho vừa với khuôn mặt
cv2.rectangle(imgTest, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 0), 2)

#so sánh 2 khuôn mặt và tìm ra khoảng cách tỉ lệ sai của mặt
result = face_recognition.compare_faces([encodeElon], encodeElonTest)
faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest)
print(result, faceDistance)
cv2.putText(imgTest, f'{result} {round(faceDistance[0], 2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

cv2.imshow('Elon Musk', imgElon)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)








