import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. Khởi tạo mô hình
model = Sequential()

# 2. Lớp Tích chập đầu tiên (đôi mắt tìm đường nét)
# 32 bộ lọc, kích thước 3x3, ảnh đầu vào 64x64x3
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))

# 3. Lớp Gộp (nén dữ liệu)
model.add(MaxPooling2D((2, 2)))

# 4. Lớp Tích chập thứ hai (tìm các đặc điểm phức tạp hơn)
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# 5. Lớp Phẳng (duỗi thẳng mảng 2D thành 1D)
model.add(Flatten())

# 6. Lớp Dày đặc (các nơ-ron suy luận)
model.add(Dense(128, activation='relu'))

# 7. Lớp Đầu ra (3 nơ-ron cho 3 loại hình, dùng Softmax để lấy xác suất)
model.add(Dense(3, activation='softmax'))

# 8. Biên dịch mô hình (Thiết lập quy tắc học)
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

from data_loading import x_train,y_train,x_vali,y_vali
# 9. Huấn luyện mô hình (Cho AI thực hành với dữ liệu)
# Giả sử bạn đã chạy file data_loading.py để có các biến x_train, y_train...
print("bat dau huan luyen...")
model.fit(x_train, y_train, epochs=10, validation_data=(x_vali, y_vali))

# 10. Lưu mô hình (Để dùng lại mà không cần huấn luyện lại)
model.save("mo_hinh_hinh_hoc.h5")
print("luu thanh cong!")