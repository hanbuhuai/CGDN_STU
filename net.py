# -*- coding:UTF-8 -*- 
'''
DCGAN 实际操作
1、数据尺寸：(64,64,3) (height,weight,chanel)
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np

'''
创建 判别器
'''
def discriminator_model():
    model = keras.Sequential()
    #第一层卷机
    model.add(keras.layers.Conv2D(
        64, #扫描64圈
        (5,5),#卷核 5，5
        padding="SAME",#输入输出一致
        input_shape=(64,64,3) #输入形状 64，64，3
    ))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.MaxPool2D(
        pool_size=(2,2),
    ))#(64,64,3,32)

    #第二层卷集
    model.add(keras.layers.Conv2D(
        128, #扫描128圈
        (5,5),#卷核 5，5
        padding="SAME",#输入输出一致
    ))#(64,64,3,128)
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.MaxPool2D(
        pool_size=(2,2),
    ))#(64,64,3,64)

    #第三层
    model.add(keras.layers.Conv2D(
        128, #扫描128圈
        (5,5),#卷核 5，5
        padding="SAME",#输入输出一致
    ))#(64,64,3,128)
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.MaxPool2D(
        pool_size=(2,2),
    ))#(64,64,3,64)
    #扁平化
    model.add(keras.layers.Flatten())#(64*64*3*64)
    model.add(keras.layers.Dense(1024))#([1024])
    model.add(keras.layers.Activation('tanh'))
    #映射成一个神经元
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('sigmoid'))
    return model
def generator_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(input_dim=100, units=1024))
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Dense(8*8*128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('tanh'))
    model.add(keras.layers.Reshape(
        target_shape=(8,8,128),
        input_shape=(8*8*128,)
    ))#形成图像
    #第一层
    model.add(keras.layers.UpSampling2D((2,2)))#(16,16,128)
    model.add(keras.layers.Conv2D(128,(5,5),padding='same'))#(16*16*128)
    model.add(keras.layers.Activation('tanh'))
    #第二层
    model.add(keras.layers.UpSampling2D((2,2)))#(32,32,128)
    model.add(keras.layers.Conv2D(128,(5,5),padding='same'))#(16*16*128)
    model.add(keras.layers.Activation('tanh'))
    #第三层
    model.add(keras.layers.UpSampling2D((2,2)))#(64,64,128)
    model.add(keras.layers.Conv2D(3,(5,5),padding='same'))#(64,64,3)
    model.add(keras.layers.Activation('tanh'))

    return model



def generator_containing_discriminator(generator, discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False  # 初始时 判别器 不可被训练
    model.add(discriminator)
    return model


