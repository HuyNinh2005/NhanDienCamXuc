import cv2
import numpy as np
from keras.models import load_model
import os

# Lấy đường dẫn thư mục hiện tại
script_dir = os.path.dirname(os.path.abspath(__file__))

# Tải mô hình và bộ phát hiện khuôn mặt
model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Định nghĩa nhãn cảm xúc
labels_dict = {0:'Tuc gian', 1:'Ghe tom', 2:'So hai', 
               3:'Hanh phuc', 4:'Binh thuong', 5:'Buon ba', 6:'Ngac nhien'}

# Đọc ảnh test
image_path = os.path.join(script_dir, "faces-small.jpg")
frame = cv2.imread(image_path)

if frame is None:
    print(f"❌ Lỗi: Không thể đọc file ảnh: {image_path}")
    print("Vui lòng kiểm tra lại đường dẫn ảnh!")
    exit(1)

# Chuyển sang ảnh xám
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Phát hiện khuôn mặt
faces = faceDetect.detectMultiScale(gray, 1.3, 3)

print(f"Đã phát hiện {len(faces)} khuôn mặt trong ảnh.")

# Xử lý từng khuôn mặt
for x,y,w,h in faces:
    # Cắt và xử lý ảnh khuôn mặt
    sub_face_img = gray[y:y+h, x:x+w]
    resized = cv2.resize(sub_face_img,(48,48))
    normalize = resized/255.0
    reshaped = np.reshape(normalize, (1, 48, 48, 1))
    
    # Dự đoán cảm xúc
    result = model.predict(reshaped)
    label = np.argmax(result, axis=1)[0]
    emotion = labels_dict[label]
    print(f"Cảm xúc nhận diện được: {emotion}")
    
    # Vẽ khung và nhãn
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
    cv2.putText(frame, emotion, (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
# Hiển thị kết quả
cv2.imshow("Nhận diện Cảm xúc", frame)
print("\nNhấn phím bất kỳ để thoát...")
cv2.waitKey(0)
cv2.destroyAllWindows()