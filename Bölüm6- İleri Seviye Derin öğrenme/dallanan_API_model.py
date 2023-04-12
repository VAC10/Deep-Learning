# -*- coding: utf-8 -*-
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate


#Giriş katmanı

visible = Input(shape=(64, 64, 1))# 64x64 1 kanallı

#Öznitelik çıkarma işlemleri için evrişim katmanı

#1. evrişim katmanı
conv1 = Conv2D(32, kernel_size = 4, activation = 'relu')(visible)# bu katmanın girişine visible'yı verdik
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)# bu işlemin girişide conv1 den gelsin
flat1 = Flatten()(pool1)

#2. evrişim katmanı
conv2 = Conv2D(16, kernel_size = 8, activation = 'relu')(visible) # 2 evrişim katmanıda aynı girişi kullanarak birbirinden bağımsız evrişim yapacak
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat2 = Flatten()(pool2)


#birleştirme katmanı

merge = concatenate([flat1, flat2])

hidden1 = Dense(10, activation='relu')(merge) #hidden layer oluşturduk 10 sınıflı tam bağlı noron oluşturduk

#çıkış katmanı

output = Dense(1, activation='sigmoid')(hidden1) # girişini hiddena bağladık
model = Model(inputs = visible, outputs = output)

# modeli özetle

model.summary()

plot_model(model, to_file='model.png')




































