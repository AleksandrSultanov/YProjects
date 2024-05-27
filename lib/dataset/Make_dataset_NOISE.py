import os

import cv2
import pandas as pd
import skimage
from google.colab import drive

drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Dataset')

# Загружаем данные
fold0 = pd.read_csv("fold_0_data.txt", sep="\t")
fold1 = pd.read_csv("fold_1_data.txt", sep="\t")
fold2 = pd.read_csv("fold_2_data.txt", sep="\t")
fold3 = pd.read_csv("fold_3_data.txt", sep="\t")
fold4 = pd.read_csv("fold_4_data.txt", sep="\t")

# Добавьте данные из всех этих файлов в один массив данных pandas и распечатайте информацию о ней.
total_data = pd.concat([fold0, fold1, fold2, fold3, fold4], ignore_index=True)
print(total_data.shape)
total_data.info()

# Цикл по всем изображениям
for row in total_data.iterrows():
    # Загружаем исходное изображение
    file_path = "Новая папка/" + row[1].user_id + "/landmark_aligned_face." + str(row[1].face_id) + "." + row[
        1].original_image
    if (os.path.exists(file_path)):
        print(file_path)
        image = skimage.io.imread(file_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Блок кода, для добавления шума
        output = skimage.util.random_noise(image, mode='gaussian', var=0.01)
        # Сохраняем новое изображение
        cv2.imwrite("Новая папка/" + row[1].user_id + "/landmark_aligned_face." + str(row[1].face_id) + ".NOISE_" + row[
            1].original_image, output)

        frame = pd.DataFrame([[row[1].user_id, "NOISE_" + row[1].original_image, row[1].face_id, row[1].age,
                               row[1].gender, row[1].x, row[1].y,
                               row[1].dx, row[1].dy, row[1].tilt_ang, row[1].fiducial_yaw_angle,
                               row[1].fiducial_score]])

        # Вносим информацию об изображении в CSV файл
        frame.to_csv('fold_1_data.txt', mode='a', header=False, sep="\t", index=False)

print("GOOD!")
