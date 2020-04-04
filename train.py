# -*- coding: UTF-8 -*-

"""
训练 DCGAN
"""

import glob,os
from PIL import Image
from scipy import misc
import numpy as np
import tensorflow as ts
#from matplotlib import pyplot as plt
from net import *
class train():
    def __init__(self):
        self.img_path = "images"
        self.img_data = self.load_imgs()
        self.out_put_path = "output"
        self.g = None
        self.d = None
        self.GAN = self.initGAN()
        self.batch_size = 100
        self.data_off_set = 0
        self.epochs = 100 #训练100圈
        self.testData = None
    '''
        数据加载&导出
    '''
    def load_imgs(self):#加载数据集
        if os.path.exists(self.img_path):
            data = []
            for img in glob.glob(os.path.join(self.img_path,"*")):
                data.append(misc.imread(img))
        else:
            raise Exception("包含所有图片的 images 文件夹不在此目录下，请添加")
        return self.imgs2data(data)
    def imgs2data(self,imgs):#清理数据
        imgs_data = np.array(imgs,dtype="float32")
        return (imgs_data-127.5)/127.5
    def data2imgs(self,img_data,key=None):#回为图片
        if not key==None:
            img_data = img_data[key]
        data  = img_data*127.5+127.5
        return data
    def outputImg(self,img_data,fname):
        Image.fromarray(img_data.astype(np.uint8)).save(os.path.join(self.out_put_path,fname))
    def train_data_batch(self,batchSize):
        st = self.data_off_set
        total = self.img_data.shape[0]
        if st>total:
            print("训练数据集合归零")
            self.data_off_set = 0
            st = self.data_off_set
        self.data_off_set = st+batchSize
        data = self.img_data[st:batchSize]
        label = np.array([1 for item in data])
        return data,label
    def train_blur_batch(self,batchSize):#混淆数据
        self.g.trainable = False
        g = self.g
        random_data = np.random.uniform(-1, 1, size=(batchSize, 100))
        data = g.predict(random_data, verbose=1)
        label = np.array([0 for item in data])
        return data,label
    '''
    加载网络
    '''
    def initGAN(self):
        d = discriminator_model()
        d_optimizero = tf.keras.optimizers.Adam(lr=0.0002,beta_1=0.5)
        d.compile(loss="binary_crossentropy",optimizer=d_optimizero)
        self.d = d
        g = generatoer_model()
        g_optimizero = tf.keras.optimizers.Adam(lr=0.0002,beta_1=0.5)
        g.compile(loss="binary_crossentropy",optimizer=g_optimizero)
        self.g = g
        GNA = generator_containing_discriminator(g,d)
        GNA.compile(loss="binary_crossentropy",optimizer=g_optimizero)
        return GNA
    def trainDiscriminator(self):#训练判别器
        self.d.trainable = True
        self.g.trainable = False
        data_t,label_t = self.train_data_batch(self.batch_size)
        data_f,label_f = self.train_blur_batch(self.batch_size)
        data = np.concatenate((data_t,data_f))
        label = np.concatenate((label_t,label_f))
        loss = self.d.train_on_batch(data, label)
        return loss
    def trainGenerater(self):
        self.d.trainable = False
        self.g.trainable = True
        random_data = np.random.uniform(-1, 1, size=(self.batch_size, 100))
        label = [1 for item in random_data]
        loss = self.GAN.train_on_batch(random_data,label)
        return loss
    def run(self):
        for e in range(0,self.epochs):
            d_loss=self.trainDiscriminator()
            msg = "[{e}]_<dloss:{loss}>".format(e=e,loss=d_loss)
            print(msg,end=';')
            g_loss=self.trainGenerater()
            msg = "[{e}]_<gloss:{loss}>".format(e=e,loss=g_loss)
            print(msg)
            if e % 10 == 9:
                self.g.save_weights("generator_weight", True)
                self.d.save_weights("discriminator_weight", True)
    def test(self):
        d_loss=self.trainDiscriminator()
        msg = "[{e}]_<dloss:{loss}>".format(e=1,loss=d_loss)
        print(msg)
        self.d.save_weights("discriminator_weight", True)
        

                
        
        
            
                
            
if __name__ == "__main__":
    hd = train()
    hd.run()
    


