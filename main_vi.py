from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os

# Đường dẫn thư mục dữ liệu
train_data_dir='data/train/'
validation_data_dir='data/test/'

# Tạo bộ sinh dữ liệu cho tập huấn luyện với augmentation
train_datagen = ImageDataGenerator(
                    rescale=1./255,  # Chuẩn hóa pixel về khoảng [0,1]
                    rotation_range=30,  # Xoay ảnh ngẫu nhiên
                    shear_range=0.3,  # Biến dạng góc
                    zoom_range=0.3,  # Phóng to/thu nhỏ
                    horizontal_flip=True,  # Lật ngang
                    fill_mode='nearest')  # Điền pixel bị thiếu

# Tạo bộ sinh dữ liệu cho tập kiểm thử (chỉ chuẩn hóa)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Cấu hình generator cho tập huấn luyện
train_generator = train_datagen.flow_from_directory(
                    train_data_dir,
                    color_mode='grayscale',  # Ảnh xám
                    target_size=(48, 48),  # Kích thước ảnh
                    batch_size=32,  # Kích thước batch
                    class_mode='categorical',  # Phân loại nhiều lớp
                    shuffle=True)  # Xáo trộn dữ liệu

# Cấu hình generator cho tập kiểm thử
validation_generator = validation_datagen.flow_from_directory(
                            validation_data_dir,
                            color_mode='grayscale',
                            target_size=(48, 48),
                            batch_size=32,
                            class_mode='categorical',
                            shuffle=True)

# Nhãn các lớp cảm xúc
class_labels=['Tuc gian','Ghe tom', 'So hai', 'Hanh phuc','Binh thuong','Buon ba','Ngac nhien']

# Lấy một mẫu dữ liệu
img, label = train_generator.__next__()

# Xây dựng mô hình CNN
model = Sequential()

# Khối Convolution thứ 1
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

# Khối Convolution thứ 2
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# Khối Convolution thứ 3
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# Khối Convolution thứ 4
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

# Làm phẳng dữ liệu
model.add(Flatten())

# Các lớp fully connected
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# Lớp output với 7 nơ-ron (7 cảm xúc)
model.add(Dense(7, activation='softmax'))

# Biên dịch mô hình
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# Đếm số lượng ảnh trong tập dữ liệu
train_path = "data/train/"
test_path = "data/test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

print(f"Số lượng ảnh huấn luyện: {num_train_imgs}")
print(f"Số lượng ảnh kiểm thử: {num_test_imgs}")

# Huấn luyện mô hình
epochs=30

print("Bắt đầu huấn luyện mô hình...")
history=model.fit(train_generator,
                steps_per_epoch=num_train_imgs//32,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=num_test_imgs//32)

# Lưu mô hình
print("Lưu mô hình...")
model.save('model_file.h5')
print("Đã lưu mô hình thành công!")