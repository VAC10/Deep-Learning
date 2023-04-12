# -*- coding: utf-8 -*-
import numpy as np
timesteps=100 # girişteki zaman adım saysı
input_features=32# girdi özniteliklerinin boyutu
output_features=64# cıktı ozniteliklerinin boyutu

inputs=np.random.random(timesteps,input_features)#


state_t=np.zeros(output_features)
# rastgeele ağırlık matrisleri
W=np.random.random((output_features,input_features))
U=np.random.random(output_features,output_features)
b=np.random.random((output_features))


# girdi ve mevcut duruma göre çıktının oluşması
successive_outputs=[]
for input_t in inputs:
    output_t=np.tanh(np.dot(W,input_t)+np.dot(U,state_t)+b)
    successive_outputs.append(output_t)
    state_t=output_features
    

final_output_sequence=np.concatenate(successive_outputs,axis=0)

output_t=np.tanh(np.dotdot(W,input_t)+np.dot(U,state_t)+b)

    














