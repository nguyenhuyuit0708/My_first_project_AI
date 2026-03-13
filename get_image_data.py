import cv2
import random
import numpy as np

#Tạo ảnh đen 
# canvas = np.zeros((64,64,3),dtype=("uint8"))

# # vẽ hình tròn lên canvas, tâm (x,y), bán kính r, màu rgb(),dày viền
# cv2.circle(canvas, (32,32),20,(255,255,255),2)
# # vẽ hình vuông lên canvas, góc trái trên (10,10), góc phải dưới(30,30), màu rgb(), dày viền
# cv2.rectangle(canvas,(10,10),(50,50),(0,255,0),2)

# pts = np.array([[32,10],[10,50],[54,50]], np.int32)# mảng các đỉnh của tam giác 
# # vẽ hình tam giác lên canvas, với các đỉnh trong mảng pts, đóng kính, màu, dày viền 

# cv2.polylines(canvas,[pts],isClosed=True,color=(0,0,255),thickness=2)


def draw_circle(n):
    for i in range(n):
        nen = np.full((64,64,3),255,dtype="uint8")
        x = random.randint(20, 44)   # Tọa độ x ngẫu nhiên từ 20 đến 44
        y = random.randint(20, 44)   # Tọa độ y ngẫu nhiên
        r = random.randint(10, 20)   # Bán kính ngẫu nhiên từ 10 đến 20
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Màu RGB ngẫu nhiên
        cv2.circle(nen,(x,y),r,color,-1)#-1 là kín hình
        filename = f"Dataset/Tron/tron_{i}.png"
        cv2.imwrite(filename,nen)

def draw_rectangle(n):
    for i in range(n):
        nen = np.full((64,64,3),255,dtype="uint8")
        up_left = [random.randint(0,32),random.randint(0,32)]
        down_right = [up_left[0]+random.randint(1,64-up_left[0]),up_left[1]+random.randint(1,64-up_left[1])]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Màu RGB ngẫu nhiên
        cv2.rectangle(nen,up_left,down_right,color,-1)
        filename = f"Dataset/Vuong/vuong_{i}.png"
        cv2.imwrite(filename,nen)
 
def draw_triangle(n):
    for i in range(n):
        nen = np.full((64,64,3),255,dtype="uint8")
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Màu RGB ngẫu nhiên
        pts=np.array([[random.randint(0, 64),random.randint(0, 64)],[random.randint(0, 64),random.randint(0, 64)],[random.randint(0, 64),random.randint(0, 64)]], np.int32)
        cv2.polylines(nen,[pts],isClosed=True,color=color,thickness=1)
        cv2.fillPoly(nen, pts=[pts], color=color)
        filename = f"Dataset/Tam_giac/tam_giac_{i}.png"
        cv2.imwrite(filename,nen) 

def make_dataset(n):
    draw_circle(n)
    draw_rectangle(n)
    draw_triangle(n)

if __name__ == "__main__":
    make_dataset(1000)