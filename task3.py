
# coding: utf-8

# In[64]:


import pandas as pd
import csv
from numpy import genfromtxt
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, UpSampling2D, Dense, Activation, Flatten
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[65]:


def featuresExtarctionFromCSV(filename):
    featureMatrix = []
    with open(filename,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            featureRow = []
            for col in row:
                featureRow.append(col)
            featureMatrix.append(featureRow)
    featureMatrix = np.delete(featureMatrix,[0],axis=0)     
    return featureMatrix

def datagenerator(inputMatrix,batchSize,filePath):
    counter = 0
    while True:
        counter =0
        batchSizeColumns = np.random.randint(0,inputMatrix.shape[0],batchSize)
        #print(len(batchSizeColumns))
        x,y,target = [],[],[]
        for i in range(0,len(batchSizeColumns)):
            dataPoint = inputMatrix[batchSizeColumns[i]]

            leftImage = dataPoint[0]
            rightImage = dataPoint[1]
            #print(leftImage)

            leftImageRead = cv2.imread(filePath +leftImage,0 )
            rightImageRead = cv2.imread(filePath +rightImage,0 )

            x.append(255.0-leftImageRead.reshape((64,64,1)))
            y.append(255.0-rightImageRead.reshape((64,64,1)))
            target.append(dataPoint[2])
            
            counter +=1
            if counter == len(batchSizeColumns):
                yield [np.array(x),np.array(y)],[np.array(target)]

def TestingDataGenerator(testingMatrix,filePath):
    x,y,target = [],[],[]
    for i in range(0,testingMatrix.shape[0]):
        dataPoint = testingMatrix[i]
        leftImage = dataPoint[0]
        rightImage = dataPoint[1]
        
        leftImageRead = cv2.imread(filePath +leftImage,0)
        rightImageRead = cv2.imread(filePath +rightImage,0 )
       
        x.append(255.0-leftImageRead.reshape((64,64,1)))
        y.append(255.0-rightImageRead.reshape((64,64,1)))
        target.append(dataPoint[2])
    return np.array(x),np.array(y),np.array(target)
       
        


# In[66]:


imDim = 64
input_shape  = (imDim,imDim,1)
inp_img = Input(shape = (imDim,imDim,1), name = 'ImageInput')
model = inp_img

#     model = Input(shape=(imDim,imDim,1))
#     model.add(Input(shape = (imDim,imDim,1), name = 'FeatureNet_ImageInput'))
model = Conv2D(32,kernel_size=(3, 3),activation='relu',input_shape=input_shape,padding='valid')(model)
#     model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model = MaxPooling2D((2,2), padding='valid')(model)
model = Conv2D(64, (3, 3), activation='relu',padding='valid')(model)
#     model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
model = MaxPooling2D((2,2),padding='valid')(model)
#     model.add(Conv2D(16, (3, 3), activation='relu',padding='same'))
model = Conv2D(128, (3, 3), activation='relu',padding='valid')(model)
model = MaxPooling2D((2,2),padding='valid')(model)
#     model.add(Conv2D(1, (3, 3), activation='relu',padding='same'))
#     model.add(Conv2D(2, (3, 3), activation='relu',padding='same'))

model = Conv2D(256, (1, 1), activation='relu',padding='valid')(model)
model = MaxPooling2D((2,2),padding='valid')(model)

model = Conv2D(64, (1, 1), activation='relu',padding='valid')(model)
# model = MaxPooling2D((2,2),padding='valid')(model)
model = Flatten()(model)

# img_in = np.array((-1,imDim,imDim,1), dtype='float32')
# img_in = tf.placeholder(shape=(imDim,imDim,1), dtype='float32')

feat = Model(inputs=[inp_img], outputs=[model],name = 'Feat_Model')
feat.summary()


# In[27]:

left_img = Input(shape = (imDim,imDim,1), name = 'left_img')
right_img = Input(shape = (imDim,imDim,1), name = 'right_img')


# In[28]:

left_feats = feat(left_img)
right_feats = feat(right_img)


# In[35]:
from keras.layers import concatenate
import random


# In[36]:

merged_feats = concatenate([left_feats, right_feats], name = 'concat_feats')
merged_feats = Dense(1024, activation = 'linear')(merged_feats)
merged_feats = BatchNormalization()(merged_feats)
merged_feats = Activation('relu')(merged_feats)
merged_feats = Dense(4, activation = 'linear')(merged_feats)
merged_feats = BatchNormalization()(merged_feats)
merged_feats = Activation('relu')(merged_feats)
merged_feats = Dense(1, activation = 'sigmoid')(merged_feats)
similarity_model = Model(inputs = [left_img, right_img], outputs = [merged_feats], name = 'Similarity_Model')

similarity_model.summary()


# In[67]:


TrainingDatasetSeen = np.array(np.delete(featuresExtarctionFromCSV("seen_Training.csv"),[0],axis=1))
print(TrainingDatasetSeen.shape)
ShuffledTrainingSeen = np.random.permutation((TrainingDatasetSeen))
print(ShuffledTrainingSeen.shape)
similarity_model.compile(optimizer = 'sgd',loss = 'binary_crossentropy',metrics=['accuracy'])
epochSize = 30
batchSize = 100
hist = []
filePath = 'Desktop/spring2019/Aml/proj2/AND_dataset/AND_images/' 

# for i in range(0,epochSize):
#     leftImage,rightImage,target = datagenerator(ShuffledTrainingSeen,batchSize,'Desktop/spring2019/Aml/proj2/AND_dataset/AND_images/')
#     history = similarity_model.fit([leftImage,rightImage],target)
#     hist.append(history.history)
history = similarity_model.fit_generator(datagenerator(ShuffledTrainingSeen,batchSize,filePath),
                                         steps_per_epoch=batchSize,epochs=10)
    


# In[68]:


# loss = []
# accr = []
# for i in range(0,len(hist)):
#     loss.append(hist[i]['loss'])
#     accr.append(hist[i]['acc'])


# In[69]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# In[72]:


TestingDatasetSeen = np.array(np.delete(featuresExtarctionFromCSV("seen_Validation.csv"),[0],axis=1))
print(TestingDatasetSeen.shape)

leftImageTest,RightImageTest,TargetTest = TestingDataGenerator(TestingDatasetSeen,'Desktop/spring2019/Aml/proj2/AND_dataset/AND_validation/')
print(leftImageTest.shape)
print(RightImageTest.shape)
print(TargetTest.shape)
output = similarity_model.predict([leftImageTest,RightImageTest])


# In[73]:


predictedOutput = []
for i in range(0,len(output)):
    if output[i]>=0.5:
        predictedOutput.append(1)
    else:
        predictedOutput.append(0)
        
right = 0
wrong = 0
for i in range(0,len(predictedOutput)):
    if int(predictedOutput[i]) == int(TargetTest[i]):
        right +=1
    else:
        wrong +=1
print(right/(right+wrong)*100)


# # shuffled dataset
# 

# In[80]:


TrainingDatasetShuffled = np.array(np.delete(featuresExtarctionFromCSV("shuffled_Training.csv"),[0],axis=1))
print(TrainingDatasetShuffled.shape)
ShuffledTrainingShuffled = np.random.permutation((TrainingDatasetShuffled))
print(ShuffledTrainingShuffled.shape)
similarity_model.compile(optimizer = 'sgd',loss = 'binary_crossentropy',metrics=['accuracy'])
epochSize = 30
batchSize = 100
hist = []
history = similarity_model.fit_generator(datagenerator(ShuffledTrainingShuffled,batchSize,filePath),
                                         steps_per_epoch=batchSize,epochs=10)


# In[81]:


# loss = []
# accr = []
# for i in range(0,len(hist)):
#     loss.append(hist[i]['loss'])
#     accr.append(hist[i]['acc'])


# In[82]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# In[83]:


# plt.plot(accr)


# In[84]:


TestingDatasetShuffled = np.array(np.delete(featuresExtarctionFromCSV("shuffled_Validation.csv"),[0],axis=1))
print(TestingDatasetShuffled.shape)

leftImageTest,RightImageTest,TargetTest = TestingDataGenerator(TestingDatasetShuffled,'Desktop/spring2019/Aml/proj2/AND_dataset/Shuffle_validation/')
print(leftImageTest.shape)
print(RightImageTest.shape)
print(TargetTest.shape)
output = similarity_model.predict([leftImageTest,RightImageTest])


# In[85]:


predictedOutput = []
for i in range(0,len(output)):
    if output[i]>=0.5:
        predictedOutput.append(1)
    else:
        predictedOutput.append(0)
        
right = 0
wrong = 0
for i in range(0,len(predictedOutput)):
    if int(predictedOutput[i]) == int(TargetTest[i]):
        right +=1
    else:
        wrong +=1
print(right/(right+wrong)*100)


# # unseen dataset

# In[88]:


TrainingDatasetUnseen = np.array(np.delete(featuresExtarctionFromCSV("unseen_Training.csv"),[0],axis=1))
print(TrainingDatasetUnseen.shape)
ShuffledTrainingUnseen = np.random.permutation((TrainingDatasetUnseen))
print(ShuffledTrainingUnseen.shape)
epochSize = 70
batchSize = 50
hist = []
# for i in range(0,epochSize):
#     leftImage,rightImage,target = datagenerator(ShuffledTrainingUnseen,batchSize,'Desktop/spring2019/Aml/proj2/AND_dataset/unseen_train/')
#     history = similarity_model.fit([leftImage,rightImage],target)
#     hist.append(history.history)
history = similarity_model.fit_generator(datagenerator(ShuffledTrainingUnseen,batchSize,filePath),
                                         steps_per_epoch=batchSize,epochs=10)


# In[89]:


import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.DataFrame(history.history)
df.plot(subplots=True, grid=True, figsize=(10,15))


# In[90]:


# import matplotlib.pyplot as plt
# plt.plot(loss)


# In[91]:


# plt.plot(accr)


# In[92]:


TestingDatasetUnseen = np.array(np.delete(featuresExtarctionFromCSV("unseen_Validation.csv"),[0],axis=1))
print(TestingDatasetUnseen.shape)

leftImageTest,RightImageTest,TargetTest = TestingDataGenerator(TestingDatasetUnseen,'Desktop/spring2019/Aml/proj2/AND_dataset/unseen_validation/')
print(leftImageTest.shape)
print(RightImageTest.shape)
print(TargetTest.shape)
output = similarity_model.predict([leftImageTest,RightImageTest])


# In[93]:


predictedOutput = []
for i in range(0,len(output)):
    if output[i]>=0.5:
        predictedOutput.append(1)
    else:
        predictedOutput.append(0)
        
right = 0
wrong = 0
for i in range(0,len(predictedOutput)):
    if int(predictedOutput[i]) == int(TargetTest[i]):
        right +=1
    else:
        wrong +=1
print(right/(right+wrong)*100)

