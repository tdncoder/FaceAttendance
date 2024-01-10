import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#Đường dẫn file của hình điểm danh
path = "ImagesAttendance"
images = []
className = []
mylist = os.listdir(path)
print(mylist)

for cl in mylist:
    #Ví dụ như trong file ImagesAttendance ở vị trí đầu tiên có file hình ảnh tên là Bill Gate 
    #Vì thế trong cl (class) chúng ta sẽ thêm vào class name để tách lấy tên và thêm vào một list mới
    currentImg = cv2.imread(f'{path}/{cl}')
    images.append(currentImg)
    className.append(os.path.splitext(cl)[0])
print(className)
    
#tạo hàm encode tất cả các hình ảnh có trong folder
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('DiemDanh.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
                                       #Ngày            Giờ  phút  giây           
            dateString = now.strftime('Ngay %x Vao luc: %H : %M : %S')
            f.writelines(f'\n{name},{dateString}')

encodeListKnown = findEncodings(images)
print("Encoding Complete!")

#Tạo video để kiểm tra

cap = cv2.VideoCapture(0)

#Encode camera

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    #Ta phải gửi hình encode tới file có vị trí hàm encode để so sánh 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    for encodeFace, faceLocation in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDistance)
        matchIndex = np.argmin(faceDistance)
# tới đây ta đã xác định được hết tỉ lệ đúng của các khuôn mặt trong folder 
# từ đây chúng ta bắt đầu tạo một cái "box" để gói gọn các giá trị, đặc điểm đó để viết tên

        if matches[matchIndex] > 0.9: 
            name = className[matchIndex].upper()
            #print(name)
#Tới đây chúng ta bắt đầu vẽ hộp nhận dạng khuôn mặt để xác nhận khuôn mặt + hiển thị tên
            y1,x2,y2,x1 = faceLocation
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4 # nếu không nhân lại tỉ lệ khung thì nó vẫn xác định được mặt
                                                    #nhưng mà cái khung nó sẽ không khớp với khuôn mặt trên webcam
                                                    #màu    #độ dày khuôn
            cv2.rectangle(img, (x1, y1), (x2, y2),(255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
            #tới đây sử dùng cùng phương pháp hiển thị vì hình chữ nhật ở phía dưới đã 
            # hiển thị tên sẵn và đẹp rồi nên sử dụng cùng phương thức
                                                                            #tỉ lệ  #màu        #độ dày
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) #vì tên đã được định dạng chuỗi nên ta không phải đổi nó 
            markAttendance(name)
        else:
            name = "Unknown"
            y1, x2, y2, x1 = faceLocation
            # nếu không nhân lại tỉ lệ khung thì nó vẫn xác định được mặt
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # nhưng mà cái khung nó sẽ không khớp với khuôn mặt trên webcam
            # màu    #độ dày khuôn
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (255, 0, 0), cv2.FILLED)
            # tới đây sử dùng cùng phương pháp hiển thị vì hình chữ nhật ở phía dưới đã
            # hiển thị tên sẵn và đẹp rồi nên sử dụng cùng phương thức
            # tỉ lệ  #màu        #độ dày
            # vì tên đã được định dạng chuỗi nên ta không phải đổi nó
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
# cap.release()
cv2.destroyAllWindows()
    
    
    #encode = face_recognition.face_encodings(img)[0] #tới đây ta có thể xác nhận được nhiều khuôn mặt cùng lúc

# Địa chỉ nhận dạng mặt
#faceLocation = face_recognition.face_locations(imgElon)[0]
#encodeElon = face_recognition.face_encodings(imgElon)[0]
# kẻ khung nhận dạng mặt xác định tọa độ XY để kẻ khung sao cho vừa với khuôn mặt
#cv2.rectangle(imgElon, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 0), 2)
# Địa chỉ nhận dạng mặt
#faceLocation = face_recognition.face_locations(imgTest)[0]
#encodeElonTest = face_recognition.face_encodings(imgTest)[0]
# kẻ khung nhận dạng mặt xác định tọa độ XY để kẻ khung sao cho vừa với khuôn mặt
#cv2.rectangle(imgTest, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (255, 0, 0), 2)

# so sánh 2 khuôn mặt và tìm ra khoảng cách tỉ lệ sai của mặt
#result = face_recognition.compare_faces([encodeElon], encodeElonTest)
#faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest)
#print(result, faceDistance)
#cv2.putText(imgTest, f'{result} {round(faceDistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)













