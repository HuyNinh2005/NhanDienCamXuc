import cv2
import numpy as np
from keras.models import load_model
import os

# Kiểm tra đường dẫn file
model_path = 'model_file_30epochs.h5'
cascade_path = 'haarcascade_frontalface_default.xml'

print(f"Thư mục làm việc hiện tại: {os.getcwd()}")
print(f"File model tồn tại: {os.path.exists(model_path)}")
print(f"File cascade tồn tại: {os.path.exists(cascade_path)}")

try:
    print("\nĐang tải mô hình...")
    model = load_model(model_path)
    print("✓ Đã tải mô hình thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải mô hình: {str(e)}")
    exit(1)

try:
    print("\nĐang tải bộ phát hiện khuôn mặt...")
    faceDetect = cv2.CascadeClassifier(cascade_path)
    print("✓ Đã tải bộ phát hiện khuôn mặt thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải bộ phát hiện khuôn mặt: {str(e)}")
    exit(1)

print("\n✓ Tất cả các thành phần đã được tải thành công!")
print("Bạn có thể chạy chương trình chính.")