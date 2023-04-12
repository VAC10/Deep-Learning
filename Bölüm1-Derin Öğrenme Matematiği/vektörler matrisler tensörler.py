import numpy as np

x=np.array([7,14,21])
d=x.ndim
print(d) # matris


from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
print(train_images.ndim)


digit=train_images[5]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
 

dizinim=train_images[7:77]
print(dizinim.shape)
dizinim=train_images[7:77,:,:]
print(dizinim.shape)





