import os,cv2
import shutil
import random
import numpy as np
cataloges = {"Tron": 0,"Vuong": 1,"Tam_giac":2}



def load_data():
    datas_train = []
    labels_train = []

    datas_vali = []
    labels_vali = []
    
    # load train
    base_train = "Dataset/Train"
    for folder_name,label in cataloges.items():
        folder_path = f"{base_train}/{folder_name}"
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file_name)
            image = cv2.imread(file_path)

            if image is not None:
                datas_train.append(image)
                labels_train.append(label)
    x_train = np.array(datas_train)/255.0
    y_train = np.array(labels_train)

    # load validation
    base_vali = "Dataset/Validation"
    for folder_name,label in cataloges.items():
        folder_path = f"{base_vali}/{folder_name}"
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file_name)
            image = cv2.imread(file_path)

            if image is not None:
                datas_vali.append(image)
                labels_vali.append(label)
    x_vali = np.array(datas_vali)/255.0
    y_vali = np.array(labels_vali)
    
    return x_train,y_train,x_vali,y_vali


def mixed_data():
    for folder_name,label in cataloges.items():
        folder_path = f"Dataset/{folder_name}" 
        files = os.listdir(folder_path)
        random.shuffle(files)
        os.makedirs(f"Dataset/Train/{folder_name}", exist_ok=True)
        os.makedirs(f"Dataset/Validation/{folder_name}", exist_ok=True)

        files_train = files[:800]
        files_validation = files[800:]

        for file_name in files_train:
            source = f"Dataset/{folder_name}/"+file_name
            destination = f"Dataset/Train/{folder_name}/" + file_name
            shutil.move(source,destination)

        for file_name in files_validation:
            source = f"Dataset/{folder_name}/"+file_name
            destination = f"Dataset/Validation/{folder_name}/" + file_name
            shutil.move(source,destination)

mixed_data()
x_train,y_train,x_vali,y_vali = load_data()


