# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 01:18:13 2020

@author: Hhaka
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


veri_inegol = pd.read_excel("data_inegol.xlsx")

train = veri_inegol.drop(["Unnamed: 0","Unnamed: 0.1"], axis=1)

label_encoder = LabelEncoder().fit(train.sonuc)
labels = label_encoder.transform(train.sonuc)
classes = list(label_encoder.classes_)



train = train.drop(["sonuc","toplam"], axis=1)
nb_features = 6
nb_classes = len(classes)

#Eğitim verilerini standartlaştırılması

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(train.values)

train =scaler.transform(train.values)


# Eğitim verisinin doğrlama işlemi
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train,y_valid =train_test_split(train, labels, test_size= 0.1)


#Verilerin kategorileştirilmesi
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_valid= to_categorical(y_valid)


# boyut düzenlemesi

X_train = np.array(X_train).reshape(1047, 6,1)
X_valid = np.array(X_valid).reshape(117, 6,1)


# Modelin oluşturulması

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Conv1D,Dropout, MaxPooling1D, Flatten


model = Sequential()
model.add(Conv1D(512,1,input_shape=(nb_features,1)))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Conv1D(512,1))
model.add(Activation("relu"))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="softmax"))
model.summary()




# Ağın derlenmesi
model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=["accuracy"])


# modelin eğitilmesi

model.fit(X_train,y_train,epochs=75, validation_data=(X_valid,y_valid))




# Ortalama Değerlerin Gösterilmesi

print("Ortalama Eğitim Kaybı :",np.mean(model.history.history["loss"]))
print("Ortalama Eğitim Başarımı :",np.mean(model.history.history["accuracy"]))
print("Ortalama Doğrulama Kaybı :",np.mean(model.history.history["val_loss"]))
print("Ortalama Doğrulama Başarımı :",np.mean(model.history.history["val_accuracy"]))




model.save("inegol.h5")











