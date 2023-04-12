# -*- coding: utf-8 -*-
from keras import layers# katman kutuphanesi
#► Katmanlar
layer=layers.Dense(32,input_shape=(784,)) # dense =noron

#Modeller
from keras import models
model=models.Sequential()
model.add(layers.Dense(32,input_shape=(784,))) # input shape = girişe bir vektör veriyoruz ve o vektörün uzunluğu
model.add(layers.Dense(32))

model.summary() # modelimizi görelim 









