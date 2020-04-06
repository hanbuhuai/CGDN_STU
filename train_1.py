# -*- coding: UTF-8 -*-
import os,glob,imageio,time
import numpy as np
from net import *
from tensorflow import keras
class Trainer():
    def __init__(self):
        #学习参数设置
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.0002
        self.BETA_1 = 0.5
        #根目录
        self.dRoot = os.path.abspath(os.path.dirname(__file__))
        #data_set 目录
        self.p_data_set = os.path.join(self.dRoot,"images")
        #out_put  目录
        self.p_output= os.path.join(self.dRoot,"output")
        #模型权重
        self.p_g_weight= os.path.join(self.dRoot,"weight/g_weight")
        self.p_d_weight= os.path.join(self.dRoot,"weight/d_weight")
        #数据[-1,64,64,3]
        self.data_set = None
        #实例模型
        self.m_g = None
        self.m_d = None
        self.m_gd =None
        self.cur_b = 0
    def loadImgs(self):#读取图片
        data = list()
        for img in glob.glob(os.path.join(self.p_data_set,"*")):
            data.append(imageio.imread(img))
        data = np.array(data)
        data = (data.astype(np.float32) - 127.5) / 127.5
        #补全
        bsize = self.BATCH_SIZE
        d_len = data.shape[0]
        padding = bsize-(d_len%bsize)
        self.data_set= np.concatenate((data,data[0:padding]))
        print("图片读取成功",data.shape)
        return self
    def data_batch(self):
        #获取真数据
        bsize = self.BATCH_SIZE
        cur_b = self.cur_b
        st = cur_b*bsize
        cur_b+=1
        ed = cur_b*bsize
        x_t = self.data_set[st:ed]
        y_t = np.ones(shape=(x_t.shape[0],1),dtype="int")
        #生成假数据
        x_f_d = np.random.uniform(-1, 1, size=(self.BATCH_SIZE, 100))
        y_f = np.zeros(shape=(x_f_d.shape[0],1),dtype="int")
        x_f = self.m_g.predict(x_f_d,verbose=0)
        x = np.concatenate((x_t,x_f))
        y = np.concatenate((y_t,y_f))
        self.cur_b = cur_b
        return x,y
    def initModel(self):#初始化模型
        #生成器配置
        m_g = generator_model()
        g_optimizer = keras.optimizers.Adam(lr=self.LEARNING_RATE, beta_1=self.BETA_1)
        m_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
        #判别器配置
        m_d = discriminator_model()
        m_d.trainable = True
        d_optimizer = keras.optimizers.Adam(lr=self.LEARNING_RATE, beta_1=self.BETA_1)
        m_d.compile(loss="binary_crossentropy", optimizer=d_optimizer)
        #总网络配置
        m_gd = generator_containing_discriminator(m_g,m_d)
        m_gd.compile(loss="binary_crossentropy", optimizer=g_optimizer)
        self.m_d,self.m_g,self.m_gd = m_d,m_g,m_gd
        return self
    def trainGenerator(self):
        #停止判定器
        self.m_d.trainable = False
        #生成噪音
        x = np.random.uniform(-1, 1, size=(self.BATCH_SIZE, 100))
        y = np.ones(shape=(x.shape[0],1),dtype="int")
        gloss=self.m_gd.train_on_batch(x,y)
        return gloss
    def trainDiscriminator(self):
        x,y = self.data_batch()
        self.m_d.trainable = True
        dloss=self.m_d.train_on_batch(x,y)
        return dloss
    def test(self):
        self.loadImgs().initModel()
        for i in range(100):
            dloss=self.trainDiscriminator()
            gloss=self.trainGenerator()
            print(dloss,gloss)

if __name__ == "__main__":
    hd = Trainer()
    hd.test()
