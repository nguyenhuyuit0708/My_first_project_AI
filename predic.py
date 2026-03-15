import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
import cv2
import numpy as np
from data_loading import cataloges
model = load_model("mo_hinh_hinh_hoc.h5")

img = cv2.imread("Datapredict/vuong_0.png")
img = cv2.resize(img,(64,64))
img = img / 255.0

# Thêm chiều batch vào đầu (axis=0)
# Biến (64, 64, 3) thành (1, 64, 64, 3)
img = np.expand_dims(img, axis=0)

ket_qua = model.predict(img)

labels = ["Hinh Tron", "Hinh Vuong", "Hinh Tam Giac"]
index = np.argmax(ket_qua)
print(f"AI du doan day la: {labels[index]}")