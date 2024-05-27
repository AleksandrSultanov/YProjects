# Импортируем библиотеки
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from google.colab import drive
from sklearn.model_selection import train_test_split

drive.mount('/content/drive')
os.chdir('/content/drive/My Drive/Dataset')


# Загружаем данные
def load_data():
    fld0 = pd.read_csv("fold_0_data.txt", sep="\t")
    fld1 = pd.read_csv("fold_1_data.txt", sep="\t")
    fld2 = pd.read_csv("fold_2_data.txt", sep="\t")
    fld3 = pd.read_csv("fold_3_data.txt", sep="\t")
    fld4 = pd.read_csv("fold_4_data.txt", sep="\t")

    # Добавьте данные из всех этих файлов в один массив данных pandas и распечатайте информацию о ней.
    total_data = pd.concat([fld1], ignore_index=True)
    print(total_data.shape)
    total_data.info()
    return total_data


# Вносим в массив данные о пути к данным
def load_path(total_data):
    imp_data = total_data[['age', 'gender', 'x', 'y', 'dx', 'dy']].copy()
    img_path = []

    for row in total_data.iterrows():
        path = "Новая папка/" + row[1].user_id + "/landmark_aligned_face." + str(row[1].face_id) + "." + row[
            1].original_image
        img_path.append(path)

    imp_data['img_path'] = img_path
    imp_data.head()
    return imp_data


# создаем карту возрастов и гендеров и сопоставляем все данные о возрасте и гендере из набора данных с возрастной группой и гендером
# удаляем записи, которые не имеют значения возраста или пола
def make_clean_data(imp_data):
    gender_to_label_map = {'f': 0, 'm': 1}

    age_to_label_map = {'0-2': 0, '4-6': 1, '8-13': 2, '15-20': 3, '25-32': 4, '38-43': 5, '48-53': 6, '60+': 7}
    age_mapping = [('(0, 2)', '0-2'), ('2', '0-2'), ('3', '0-2'), ('(4, 6)', '4-6'), ('(8, 12)', '8-13'),
                   ('13', '8-13'), ('22', '15-20'), ('(8, 23)', '15-20'), ('23', '25-32'), ('(15, 20)', '15-20'),
                   ('(25, 32)', '25-32'), ('(27, 32)', '25-32'), ('32', '25-32'), ('34', '25-32'), ('29', '25-32'),
                   ('(38, 42)', '38-43'), ('35', '38-43'), ('36', '38-43'), ('42', '48-53'), ('45', '38-43'),
                   ('(38, 43)', '38-43'), ('(38, 42)', '38-43'), ('(38, 48)', '48-53'), ('46', '48-53'),
                   ('(48, 53)', '48-53'), ('55', '48-53'), ('56', '48-53'), ('(60, 100)', '60+'), ('57', '60+'),
                   ('58', '60+')]

    age_mapping_dict = {each[0]: each[1] for each in age_mapping}
    drop_labels = []
    for idx, each in enumerate(imp_data.age):
        if each == 'None':
            drop_labels.append(idx)
        else:
            imp_data.age.loc[idx] = age_mapping_dict[each]

    imp_data = imp_data.drop(labels=drop_labels, axis=0)
    imp_data.age.value_counts(dropna=False)

    # удалить записи, c неизвестным гендером и распечатать статистику для оставшихся данных
    imp_data = imp_data.dropna()
    clean_data = imp_data[imp_data.gender != 'u'].copy()
    clean_data.info()

    # сопоствляем пол
    clean_data['gender'] = clean_data['gender'].apply(lambda g: gender_to_label_map[g])

    # сопоствляем возраст
    clean_data['age'] = clean_data['age'].apply(lambda age: age_to_label_map[age])
    return clean_data


# Разделим данные с помощью метода train_test_split из библиотеки sklearn и внесем их в соответсвующие наборы
def make_train_test(attribute, clean_data):
    X = clean_data[['img_path']]
    y = clean_data[[attribute]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print('Train data shape {}'.format(X_train.shape))
    print('Test data shape {}'.format(X_test.shape))

    train_images = []
    test_images = []

    for row in X_train.iterrows():
        image = Image.open(row[1].img_path)  # Загружаем изображение
        image = image.resize((300, 300))  # Изменим размер изображения
        data = np.asarray(image)  # Преобразуем изображение в массив
        train_images.append(data)  # Добавим в обучающий набор данных

    for row in X_test.iterrows():
        image = Image.open(row[1].img_path)  # Загружаем изображение
        image = image.resize((300, 300))  # Изменим размер изображения
        data = np.asarray(image)  # Преобразуем изображение в массив
        test_images.append(data)  # Добавим в тестовый набор данных

    train_images = np.asarray(train_images)
    test_images = np.asarray(test_images)

    print('Train images shape {}'.format(train_images.shape))
    print('Test images shape {}'.format(test_images.shape))

    return train_images, y_train, test_images, y_test


# Модель для определения пола
def gender_model(train_images, y_train, test_images, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, input_shape=(300, 300, 3), kernel_size=(7, 7), strides=3, padding='valid',
                               activation='relu'),
        # Слой 2D свертки
        tf.keras.layers.Dropout(0.2),  # Применяем Dropout к входным данным
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # Операция объединения Max для 2D пространственных данных
        tf.keras.layers.BatchNormalization(),
        # Cлой нормализует выходные данные, используя среднее значение и стандартное отклонение текущего пакета входных данных
        tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='same', activation='relu'),
        # Слой 2D свертки
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # Операция объединения Max для 2D пространственных данных
        tf.keras.layers.Dropout(0.2),  # Применяем Dropout к входным данным
        tf.keras.layers.BatchNormalization(),
        # Cлой нормализует выходные данные, используя среднее значение и стандартное отклонение текущего пакета входных данных
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
        # Слой 2D свертки
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        # Операция объединения Max для 2D пространственных данных
        tf.keras.layers.BatchNormalization(),
        # Cлой нормализует выходные данные, используя среднее значение и стандартное отклонение текущего пакета входных данных
        tf.keras.layers.Flatten(),  # Выравниваем входные данные
        tf.keras.layers.Dropout(0.2),  # Применяем Dropout к входным данным
        tf.keras.layers.Dense(units=256, activation='relu'),  # Выравниваем ввод
        tf.keras.layers.Dropout(rate=0.25),  # Применяем Dropout к входным данным
        tf.keras.layers.BatchNormalization(),
        # Cлой нормализует выходные данные, используя среднее значение и стандартное отклонение текущего пакета входных данных
        tf.keras.layers.Dense(units=128, activation='relu'),  # Плотно связанный слой нейросети
        tf.keras.layers.Dropout(rate=0.25),  # Применяем Dropout к входным данным
        tf.keras.layers.BatchNormalization(),
        # Cлой нормализует выходные данные, используя среднее значение и стандартное отклонение текущего пакета входных данных
        tf.keras.layers.Dense(units=64, activation='relu'),  # Плотно связанный слой нейросети
        tf.keras.layers.Dropout(rate=0.25),  # Применяем Dropout к входным данным
        tf.keras.layers.BatchNormalization(),
        # Cлой нормализует выходные данные, используя среднее значение и стандартное отклонение текущего пакета входных данных
        tf.keras.layers.Dense(units=16, activation='relu'),  # Плотно связанный слой нейросети
        tf.keras.layers.Dropout(rate=0.25),  # Применяем Dropout к входным данным
        tf.keras.layers.BatchNormalization(),
        # Cлой нормализует выходные данные, используя среднее значение и стандартное отклонение текущего пакета входных данных
        tf.keras.layers.Dense(units=2, activation='softmax')])  # Плотно связанный слой нейросети

    model.summary()  # Вывести информацию о структуре нейросети

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)  # Callback для остановки обучения

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])  # Компилируем модель

    history = model.fit(train_images, y_train, batch_size=32, epochs=32, validation_data=(test_images, y_test),
                        callbacks=[callback])  # Обучаем модель

    tf.keras.models.save_model(model, 'gm_BW')  # Сохраняем модель

    test_loss, test_acc = model.evaluate(test_images, y_test, verbose='auto')
    print(test_acc)  # Выводим точность

    # Выводим графики точности и потерь
    drow_graph(history)


# Модель для определения возраста
def age_model(train_images, y_train, test_images, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=64, input_shape=(300, 300, 3), kernel_size=(7, 7), strides=3, padding='valid',
                               activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(rate=0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dropout(rate=0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=8, activation='softmax')])

    model.summary()

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(train_images, y_train, batch_size=32, epochs=32, validation_data=(test_images, y_test),
                        callbacks=[callback])

    tf.keras.models.save_model(model, 'am_BW')

    test_loss, test_acc = model.evaluate(test_images, y_test, verbose=2)
    print(test_acc)

    drow_graph(history)


# Выводим графики точности и потерь
def drow_graph(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    total_data = load_data()
    imp_data = load_path(total_data)
    clean_data = make_clean_data(imp_data)
    train_images, y_train, test_images, y_test = make_train_test('gender', clean_data)
    gender_model(train_images, y_train, test_images, y_test)
    train_images, y_train, test_images, y_test = make_train_test('age', clean_data)
    age_model(train_images, y_train, test_images, y_test)
