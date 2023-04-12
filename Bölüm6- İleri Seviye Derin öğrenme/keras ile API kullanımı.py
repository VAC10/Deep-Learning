# -*- coding: utf-8 -*-
from keras import Input,layers
input_tensor=Input((32,))
dense = layers.Dense(32, activation='relu')
output_tensor = dense(input_tensor)
from keras.models import Sequential, Model
#Sequential Model
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation ='relu', input_shape=(64, ))) # 32 kanallı dense ve girişi 64'tür
seq_model.add(layers.Dense(32, activation ='relu'))
seq_model.add(layers.Dense(10, activation ='sigmoid')) # sınıflandırıcı katman çıkış katmanı

seq_model.summary()


#functional Model
input_tensor = Input(shape = (64,))

x = layers.Dense(32, activation='relu')(input_tensor)

x =layers.Dense(32, activation='relu')(x)

output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)# yukarıdakı modelden farkı modeladd diye ilerlemiyoruz, adım adım ilerleyerek en son inputla outputu bağlıyoruz

model.summary()

#Modelin derlenmesi
model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy')
#Eğitim için rastgele bir küme oluşturmak

import numpy as np
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))


#Modelin Eğitilmesi
model.fit(x_train,y_train,epochs=10,batch_size=128)

score=model.evaluate(x_train,y_train)  # değerlendirme skoru











