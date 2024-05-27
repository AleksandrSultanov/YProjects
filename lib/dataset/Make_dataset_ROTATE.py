import os

import pandas as pd
from PIL import Image
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
    image = Image.open(
        "Новая папка/" + row[1].user_id + "/landmark_aligned_face." + str(row[1].face_id) + "." + row[1].original_image)

    # Повернуть изображение на 30 и -30 градусов и сдвинуть центр
    im_rotate30 = image.rotate(30, center=(0, 0))
    im_rotate_30 = image.rotate(-30, center=(-40, -40))

    # Сохраняем новые изображения
    im_rotate30.save(
        "Новая папка/" + row[1].user_id + "/landmark_aligned_face." + str(row[1].face_id) + ".ROTATE30_" + row[
            1].original_image)
    im_rotate_30.save(
        "Новая папка/" + row[1].user_id + "/landmark_aligned_face." + str(row[1].face_id) + ".ROTATE_30_" + row[
            1].original_image)

    # Вносим информацию о них в CSV файл
    frame = pd.DataFrame([[row[1].user_id, "ROTATE30_" + row[1].original_image, row[1].face_id, row[1].age,
                           row[1].gender, row[1].x, row[1].y,
                           row[1].dx, row[1].dy, row[1].tilt_ang, row[1].fiducial_yaw_angle, row[1].fiducial_score]])
    frame.to_csv('fold_1_data.txt', mode='a', header=False, sep="\t", index=False)

    frame = pd.DataFrame([[row[1].user_id, "ROTATE_30_" + row[1].original_image, row[1].face_id, row[1].age,
                           row[1].gender, row[1].x, row[1].y,
                           row[1].dx, row[1].dy, row[1].tilt_ang, row[1].fiducial_yaw_angle, row[1].fiducial_score]])
    frame.to_csv('fold_1_data.txt', mode='a', header=False, sep="\t", index=False)

print("GOOD!")
