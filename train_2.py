# -*- coding: UTF-8 -*-
import os,glob,imageio,time
import numpy as np
from net import *
from tensorflow import keras

class train():
    def __init__(self):
        #学习参数设置
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.0002
        self.BETA_1 = 0.5
        #文件地址
        self.dRoot = os.path.dirname(os.path.abspath(__file__))
        #数据集地址
        self.imgsPath = os.path.join(self.dRoot,'images')
        self.imgsArray= np.array([])
        self.bCur = 0
        #权重生成
        self.generate_weight = os.path.join(self.dRoot,'generate_weight')
        self.discriminate_weight = os.path.join(self.dRoot,'discriminate_weight')
        self.generate_img_path = os.path.join(self.dRoot,"output/{path}/fname.png")
    """读取图片数据"""
    def __read_imgs(self):
        imgList = []
        for img in glob.glob(self.imgsPath+"/*"):
            imgList.append(imageio.imread(img))
        return np.array(imgList,dtype='float32')
    def __img_normalization(self,imgArray):
        return imgArray/127.5-1.0
    def next_batch(self):
        #不存在的话读取
        if  self.imgsArray.shape[0]==0:
            print("读取数据")
            imgArray = self.__read_imgs()
            self.imgsArray = self.__img_normalization(imgArray)
            ial = self.imgsArray.shape[0]
            padding_len = int(self.BATCH_SIZE-ial%self.BATCH_SIZE)
            self.imgsArray = np.concatenate((self.imgsArray,self.imgsArray[0:padding_len]))
        ial = self.imgsArray.shape[0]
        bCur = self.bCur
        bCur = 0 if bCur*self.BATCH_SIZE>=ial else bCur
        slice_data = [bCur*self.BATCH_SIZE,(bCur+1)*self.BATCH_SIZE]
        input_batch = self.imgsArray[slice_data[0]:slice_data[1]]
        label = [[1] for item in range(input_batch.shape[0])]
        label = np.array(label)
        self.bCur = bCur+1
        #生成干扰数据
        random_data = np.random.uniform(-1, 1, size=(self.BATCH_SIZE, 100))
        # 生成器 生成的图片数据
        generated_images = self.g.predict(random_data, verbose=0)
        generated_images = (generated_images-127.5)/127.5
        generated_labels = [[0] for item in range(input_batch.shape[0])]
        generated_labels = np.array(generated_labels)
        return np.concatenate((input_batch,generated_images)),np.concatenate((label,generated_labels))
    '''
        创建训模型
    '''
    def initModel(self):
        #创建生成器
        self.g = generator_model()
        g_optimizer = keras.optimizers.Adam(lr=self.LEARNING_RATE, beta_1=self.BETA_1)
        self.g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
        #创建判别器
        self.d = discriminator_model()
        d_optimizer = keras.optimizers.Adam(lr=self.LEARNING_RATE, beta_1=self.BETA_1)
        self.d.compile(loss="binary_crossentropy", optimizer=d_optimizer)
        #创建网络
        self.g_d = generator_containing_discriminator(self.g,self.d)
        self.g_d.compile(loss="binary_crossentropy", optimizer=g_optimizer)
        print("模型初始化成功")
        return self
    def trainDiscriminator(self,x,label):
        self.g.trainable = False
        self.d.trainable = True
        return self.d.train_on_batch(x,label)
    def trainGenerator(self,x,label):
        self.d.trainable = False
        self.g.trainable = True
        return self.g_d.train_on_batch(x,label)
    def load_weight(self):
        if os.path.isfile(self.generate_weight):
            self.g.load_weights(self.generate_weight)
        if os.path.isfile(self.discriminate_weight):
            self.d.load_weights(self.discriminate_weight)
        return self
    def save_weight(self):
        self.g.save_weights(self.generate_weight,True)
        self.d.save_weights(self.discriminate_weight,True)
        return self
    def test(self):
        pass
    def run(self,epochs,cur=1):
        for i in range(self.BATCH_SIZE):
            #训练判别器
            x,label = self.next_batch()
            d_loss = self.trainDiscriminator(x,label)
            #训练生成器
            random_data = np.random.uniform(-1, 1, size=(self.BATCH_SIZE, 100))
            label = [[1] for i in range(self.BATCH_SIZE)]
            label = np.array(label) 
            g_loss = self.trainGenerator(random_data,label)
            print("第{}轮{} 步, 生成器的损失: {:.3f}, 判别器的损失: {:.3f}".format(cur,i, g_loss, d_loss))         
            x,label = self.next_batch()
        if cur%10==1:
            self.save_weight()
        #停止条件
        stop_sign = False
        if 'endTime' in epochs:
            stop_sign = time.time>epochs['endTime']
        if 'round' in epochs:
            stop_sign = cur>round
        if stop_sign:
            self.save_weight()
            return "训练完成"
        else:
            self.run(epochs,cur+1)
        
if __name__ == "__main__":
    hd = train().initModel().load_weight().test()
    


    