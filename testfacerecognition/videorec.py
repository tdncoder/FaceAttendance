import cv2
import face_recognition

#ma hoa khuon mat tu camera
img=face_recognition.load_image_file("./image/NguyenMinhThang.png")
sfr = face_recognition.face_encodings(img)

#SỬ DỤNG TẬP TIN CÓ "CÀI ĐẶT SẴN TÊN NHẬN DẠNG" ĐỂ KHI SHOW CAMERA LÊN THÌ AI NHẬN RA ĐƯỢC LÀ AI

#sfr.load_encoding_images("./image/") # chỗ này có thể thay đổi thư mục để chèn hình khác vào có 1 cái file riêng toàn hình mặt sinh viên + mssv
# Phạm xuân đài - 20xxxxx ví dụ
#load camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    #xác nhân khuôn mặt
    face_locations, face_names = sfr.detect_known_faces(frame)
    for diachimat, ten in zip(face_locations, face_names):
        y1, x2, y2, x1 = diachimat[0], diachimat[1], diachimat[2], diachimat[3]
        #set toa độ cho khuôn mặt và tên
        cv2.putText(frame, ten, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
    
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()