# -*- coding: utf-8 -*-
# proje: mnsit veri kumesi ile rakam sınıflandırma
# mnist veri kumesini yukleme
from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
len(train_images)
len(test_images)

# Yapay Sinir  Ağı mimarisi

from keras import models
from keras import layers

network=models.Sequential() # bir boş model oluşturduk
network.add(layers.Dense(512,activation="relu",input_shape=(28*28,))) # layer=katman, dense=sinir ağı
network.add(layers.Dense(10,activation="softmax"))# bu satır sınıflayıcı katmanımız 10 sınıfımız var

network.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"]) # loss ile yitim fonksiyonu yani başarıya giderken her seferinde geriye yayılım yaparken yitim hesaplıyorz  accuracy başarı metriğidir

#girişlerin ve etiketlerin hazırlanması
# girişler:
train_images=train_images.reshape((60000,28*28))
train_images =train_images.astype("float32")/255

test_images=test_images.reshape((10000,28*28))
test_images =test_images.astype("float32")/255

#etiketler:
from keras.utils import to_categorical

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
               
# modelin eğitilmesi

network.fit(train_images,train_labels,epochs=5,batch_size=128)

test_loss,test_acc=network.evaluate(test_images,test_labels)# test loss ve test accuracy i hesapladık
print("test_loss:",test_loss)
print("test_acc:",test_acc)