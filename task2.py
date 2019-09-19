
# coding: utf-8

# In[286]:


import pandas as pd
import csv
from numpy import genfromtxt
import numpy as np
from itertools import combinations_with_replacement
import pandas as pd
from pgmpy.estimators import K2Score,ExhaustiveSearch,HillClimbSearch,ConstraintBasedEstimator
from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination


# In[287]:


# funtion for usage


# In[358]:


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

def processingDataset(pairValues):
    fetureSpecifiedPairs = []
    for i in range(0,pairValues.shape[0]):
        temp = pairValues[i]
        Image1feature = featureDict[temp[0]]
        Image2feature = featureDict[temp[1]]
        subFeature = []
       # subFeature.append(i)
        for j in range(0,len(Image1feature)):
              subFeature.append((Image1feature[j]))
        for k in range(0,len(Image2feature)):
            subFeature.append((Image2feature[k]))
        subFeature.append(int(temp[2]))
        fetureSpecifiedPairs.append(subFeature)
    return fetureSpecifiedPairs

# removing the unwanted dataset points
def removeUnwanted(TrainingDatasetSeen):
    TariningDatasetSeenDupRem = []
    for i in range(0,TrainingDatasetSeen.shape[0]):
        temp = TrainingDatasetSeen[i]
        if temp[0] in featureList1 and temp[1] in featureList1:
            TariningDatasetSeenDupRem.append(TrainingDatasetSeen[i])
    return TariningDatasetSeenDupRem

def predictionFunction(testingFetureValuesSeen,infer):
    predictedValue = []

    for i in range(0,testingFetureValuesSeen.shape[0]):
        print(i)
        temp = infer.map_query(['x'], evidence={'a': int(testingFetureValuesSeen[i][0])-1, 
                                         'b': int(testingFetureValuesSeen[i][1])-1, 
                                         'c': int(testingFetureValuesSeen[i][2])-1, 
                                         'd': int(testingFetureValuesSeen[i][3])-1, 
                                         'e': int(testingFetureValuesSeen[i][4])-1, 
                                         'f': int(testingFetureValuesSeen[i][5])-1, 
                                         'g': int(testingFetureValuesSeen[i][6])-1, 
                                         'h': int(testingFetureValuesSeen[i][7])-1, 
                                         'i': int(testingFetureValuesSeen[i][8])-1, 
                                         'j': int(testingFetureValuesSeen[i][9])-1, 
                                         'k': int(testingFetureValuesSeen[i][10])-1, 
                                         'l': int(testingFetureValuesSeen[i][11])-1, 
                                         'm': int(testingFetureValuesSeen[i][12])-1, 
                                         'n': int(testingFetureValuesSeen[i][13])-1, 
                                         'o': int(testingFetureValuesSeen[i][14])-1,
                                         'a1': int(testingFetureValuesSeen[i][15])-1, 
                                         'b1': int(testingFetureValuesSeen[i][16])-1, 
                                         'c1': int(testingFetureValuesSeen[i][17])-1,
                                         'd1': int(testingFetureValuesSeen[i][18])-1, 
                                         'e1': int(testingFetureValuesSeen[i][19])-1, 
                                         'f1': int(testingFetureValuesSeen[i][20])-1, 
                                         'g1': int(testingFetureValuesSeen[i][21])-1, 
                                         'h1': int(testingFetureValuesSeen[i][22])-1, 
                                         'i1': int(testingFetureValuesSeen[i][23])-1, 
                                         'j1': int(testingFetureValuesSeen[i][24])-1, 
                                         'k1': int(testingFetureValuesSeen[i][25])-1, 
                                         'l1': int(testingFetureValuesSeen[i][26])-1, 
                                         'm1': int(testingFetureValuesSeen[i][27])-1, 
                                         'n1': int(testingFetureValuesSeen[i][28])-1, 
                                         'o1': int(testingFetureValuesSeen[i][29])-1})

        predictedValue.append(temp)
    return predictedValue


def accuracy(targetValuesSeen,predictedValue):
    
    finalPredValues = []
    for i in range(0,len(predictedValue)):
        temp = predictedValue[i]['x']
        finalPredValues.append(temp)

    right = 0
    wrong = 0
    for i in range(0,len(finalPredValues)):
        if int(finalPredValues[i]) == int(targetValuesSeen[i]):
            right +=1
        else:
            wrong +=1
    return str(right/(right+wrong)*100)


# In[289]:


# extracting 15 features from csv file
FeatureMatrix = np.array(featuresExtarctionFromCSV("15features.csv"))
print(FeatureMatrix.shape)

# passing the feature values into dictionary
featureDict = dict()
for i in range(0,len(FeatureMatrix)):
    temp = FeatureMatrix[i]
    featureDict[temp[0]] = temp[1:18]
    

featureList1 = np.transpose(FeatureMatrix)[0]

# output = list(combinations_with_replacement(featureList1, 2))
# print(output)


# # seen dataset

# In[290]:


# extracting the training data from csv
TrainingDatasetSeen = np.array(np.delete(featuresExtarctionFromCSV("seen_Training.csv"),[0],axis=1))
ValidationDatasetSeen = np.array(np.delete(featuresExtarctionFromCSV("seen_Validation.csv"),[0],axis=1))
print(TrainingDatasetSeen.shape)
print(ValidationDatasetSeen.shape)

TariningDatasetSeenDupRem = np.array(removeUnwanted(TrainingDatasetSeen))
print(TariningDatasetSeenDupRem.shape)


# In[291]:


TestingDatasetSeenDupRem = np.array(removeUnwanted(ValidationDatasetSeen))
print(TestingDatasetSeenDupRem.shape)


# In[292]:


# removing the image ids from FeatureMatrix
featureMatrixImageIdRem = np.delete(FeatureMatrix,[0],axis=1)
print(featureMatrixImageIdRem.shape)


# # hill climb search for best model estimate

# In[306]:


col = list('abcdefghijklmno')
dataset = pd.DataFrame(featureMatrixImageIdRem, columns = col)
est = HillClimbSearch(dataset,scoring_method = K2Score(dataset))
bestModel = est.estimate()#max_indegree=2)
print(bestModel.edges())


# In[340]:


model = BayesianModel([('a', 'e'), ('a', 'b'), ('c', 'g'), ('c', 'a'), 
                      ('c', 'l'), ('c', 'b'), ('c', 'm'), ('c', 'i'), 
                      ('d', 'c'), ('d', 'f'), ('d', 'g'), ('d', 'a'), 
                      ('e', 'j'), ('e', 'm'), ('f', 'm'), ('f', 'b'),
                      ('f', 'i'), ('f', 'j'), ('f', 'e'), ('g', 'f'), 
                      ('g', 'h'), ('i', 'a'), ('k', 'o'), ('k', 'n'),
                      ('k', 'd'), ('k', 'l'), ('k', 'f'), ('k', 'c'), 
                      ('k', 'j'), ('l', 'f'), ('l', 'm'), ('l', 'e'), 
                      ('l', 'g'), ('l', 'i'), ('n', 'd'), ('n', 'j'),
                      ('n', 'c'), ('n', 'l'), ('n', 'o'), ('o', 'j'), ('o', 'b'), ('o', 'd'),
                      ('a1', 'e1'), ('a1', 'b1'), ('c1', 'g1'), ('c1', 'a1'), 
                      ('c1', 'l1'), ('c1', 'b1'), ('c1', 'm1'), ('c1', 'i1'), 
                      ('d1', 'c1'), ('d1', 'f1'), ('d1', 'g1'), ('d1', 'a1'), 
                      ('e1', 'j1'), ('e1', 'm1'), ('f1', 'm1'), ('f1', 'b1'),
                      ('f1', 'i1'), ('f1', 'j1'), ('f1', 'e1'), ('g1', 'f1'), 
                      ('g1', 'h1'), ('i1', 'a1'), ('k1', 'o1'), ('k1', 'n1'),
                      ('k1', 'd1'), ('k1', 'l1'), ('k1', 'f1'), ('k1', 'c1'), 
                      ('k1', 'j1'), ('l1', 'f1'), ('l1', 'm1'), ('l1', 'e1'), 
                      ('l1', 'g1'), ('l1', 'i1'), ('n1', 'd1'), ('n1', 'j1'),
                      ('n1', 'c1'), ('n1', 'l1'), ('n1', 'o1'), ('o1', 'j1'), ('o1', 'b1'), ('o1', 'd1'),
                      ('a','x'),('f','x'),('f1','x'),('a1','x'),('b','x'),('b1','x')])


# # training the Bayesian model

# In[341]:


trainingDatasetFinal = processingDataset(TariningDatasetSeenDupRem)


# In[342]:


col2 = list(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
            'a1','b1','c1','d1','e1','f1','g1','h1','i1','j1','k1','l1','m1','n1','o1','x'])
concatDataset = pd.DataFrame(trainingDatasetFinal, columns = col2)


# In[343]:


model.fit(concatDataset)


# # validating the dataset/testing the model

# In[344]:


testingDatasetFinal = np.array(processingDataset(TestingDatasetSeenDupRem))


# In[345]:


print(testingDatasetFinal)

testingFetureValuesSeen =  np.transpose(np.transpose(testingDatasetFinal)[0:30])
print(testingFetureValuesSeen.shape)
inferSeen = VariableElimination(model)


# In[346]:


PredictedValuesSeen = predictionFunction(testingFetureValuesSeen,inferSeen)


# In[347]:


targetValuesSeen = np.transpose(np.transpose(testingDatasetFinal)[30:31])
print("accuracy: " + accuracy(targetValuesSeen,PredictedValuesSeen)


# # shuffled dataset

# In[349]:


TrainingDatasetShuffled = np.array(np.delete(featuresExtarctionFromCSV("shuffled_Training.csv"),[0],axis=1))
ValidationDatasetShuffled = np.array(np.delete(featuresExtarctionFromCSV("shuffled_Validation.csv"),[0],axis=1))
print(TrainingDatasetShuffled.shape)
print(ValidationDatasetShuffled.shape)

TariningDatasetShuffDupRem = np.array(removeUnwanted(TrainingDatasetShuffled))
print(TariningDatasetShuffDupRem.shape)

TestingDatasetShuffDupRem = np.array(removeUnwanted(ValidationDatasetShuffled))
print(TestingDatasetShuffDupRem.shape)


# In[350]:


model = BayesianModel([('a', 'e'), ('a', 'b'), ('c', 'g'), ('c', 'a'), 
                      ('c', 'l'), ('c', 'b'), ('c', 'm'), ('c', 'i'), 
                      ('d', 'c'), ('d', 'f'), ('d', 'g'), ('d', 'a'), 
                      ('e', 'j'), ('e', 'm'), ('f', 'm'), ('f', 'b'),
                      ('f', 'i'), ('f', 'j'), ('f', 'e'), ('g', 'f'), 
                      ('g', 'h'), ('i', 'a'), ('k', 'o'), ('k', 'n'),
                      ('k', 'd'), ('k', 'l'), ('k', 'f'), ('k', 'c'), 
                      ('k', 'j'), ('l', 'f'), ('l', 'm'), ('l', 'e'), 
                      ('l', 'g'), ('l', 'i'), ('n', 'd'), ('n', 'j'),
                      ('n', 'c'), ('n', 'l'), ('n', 'o'), ('o', 'j'), ('o', 'b'), ('o', 'd'),
                      ('a1', 'e1'), ('a1', 'b1'), ('c1', 'g1'), ('c1', 'a1'), 
                      ('c1', 'l1'), ('c1', 'b1'), ('c1', 'm1'), ('c1', 'i1'), 
                      ('d1', 'c1'), ('d1', 'f1'), ('d1', 'g1'), ('d1', 'a1'), 
                      ('e1', 'j1'), ('e1', 'm1'), ('f1', 'm1'), ('f1', 'b1'),
                      ('f1', 'i1'), ('f1', 'j1'), ('f1', 'e1'), ('g1', 'f1'), 
                      ('g1', 'h1'), ('i1', 'a1'), ('k1', 'o1'), ('k1', 'n1'),
                      ('k1', 'd1'), ('k1', 'l1'), ('k1', 'f1'), ('k1', 'c1'), 
                      ('k1', 'j1'), ('l1', 'f1'), ('l1', 'm1'), ('l1', 'e1'), 
                      ('l1', 'g1'), ('l1', 'i1'), ('n1', 'd1'), ('n1', 'j1'),
                      ('n1', 'c1'), ('n1', 'l1'), ('n1', 'o1'), ('o1', 'j1'), ('o1', 'b1'), ('o1', 'd1'),
                      ('a','x'),('f','x'),('f1','x'),('a1','x'),('b','x'),('b1','x')])


# In[351]:


trainingDatasetFinalShuffle = processingDataset(TariningDatasetShuffDupRem)


# In[352]:


col2 = list(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
            'a1','b1','c1','d1','e1','f1','g1','h1','i1','j1','k1','l1','m1','n1','o1','x'])
concatDatasetShuffle = pd.DataFrame(trainingDatasetFinalShuffle, columns = col2)


# In[353]:


model.fit(concatDatasetShuffle)


# In[354]:


# testing o shuffled dataset


# In[355]:


testingDatasetFinalShuffle = np.array(processingDataset(TestingDatasetShuffDupRem))


# In[359]:


print(testingDatasetFinal)

testingFetureValuesShuffle =  np.transpose(np.transpose(testingDatasetFinalShuffle)[0:30])
print(testingFetureValuesShuffle.shape)
infer = VariableElimination(model)


# In[360]:


PredictedValuesShuffle = predictionFunction(testingFetureValuesShuffle,infer)


# In[365]:


targetValue = np.transpose(np.transpose(testingDatasetFinalShuffle)[30:31])
print("accuracy: " + accuracy(targetValue,PredictedValuesShuffle))


# # unseen dataset

# In[367]:


TrainingDatasetUnseen = np.array(np.delete(featuresExtarctionFromCSV("unseen_Training.csv"),[0],axis=1))
ValidationDatasetUnseen = np.array(np.delete(featuresExtarctionFromCSV("unseen_Validation.csv"),[0],axis=1))
print(TrainingDatasetUnseen.shape)
print(ValidationDatasetUnseen.shape)

TariningDatasetUnseenDupRem = np.array(removeUnwanted(TrainingDatasetUnseen))
print(TariningDatasetUnseenDupRem.shape)

TestingDatasetUnseenDupRem = np.array(removeUnwanted(ValidationDatasetUnseen))
print(TestingDatasetUnseenDupRem.shape)


# In[368]:


modelUnseen = BayesianModel([('a', 'e'), ('a', 'b'), ('c', 'g'), ('c', 'a'), 
                      ('c', 'l'), ('c', 'b'), ('c', 'm'), ('c', 'i'), 
                      ('d', 'c'), ('d', 'f'), ('d', 'g'), ('d', 'a'), 
                      ('e', 'j'), ('e', 'm'), ('f', 'm'), ('f', 'b'),
                      ('f', 'i'), ('f', 'j'), ('f', 'e'), ('g', 'f'), 
                      ('g', 'h'), ('i', 'a'), ('k', 'o'), ('k', 'n'),
                      ('k', 'd'), ('k', 'l'), ('k', 'f'), ('k', 'c'), 
                      ('k', 'j'), ('l', 'f'), ('l', 'm'), ('l', 'e'), 
                      ('l', 'g'), ('l', 'i'), ('n', 'd'), ('n', 'j'),
                      ('n', 'c'), ('n', 'l'), ('n', 'o'), ('o', 'j'), ('o', 'b'), ('o', 'd'),
                      ('a1', 'e1'), ('a1', 'b1'), ('c1', 'g1'), ('c1', 'a1'), 
                      ('c1', 'l1'), ('c1', 'b1'), ('c1', 'm1'), ('c1', 'i1'), 
                      ('d1', 'c1'), ('d1', 'f1'), ('d1', 'g1'), ('d1', 'a1'), 
                      ('e1', 'j1'), ('e1', 'm1'), ('f1', 'm1'), ('f1', 'b1'),
                      ('f1', 'i1'), ('f1', 'j1'), ('f1', 'e1'), ('g1', 'f1'), 
                      ('g1', 'h1'), ('i1', 'a1'), ('k1', 'o1'), ('k1', 'n1'),
                      ('k1', 'd1'), ('k1', 'l1'), ('k1', 'f1'), ('k1', 'c1'), 
                      ('k1', 'j1'), ('l1', 'f1'), ('l1', 'm1'), ('l1', 'e1'), 
                      ('l1', 'g1'), ('l1', 'i1'), ('n1', 'd1'), ('n1', 'j1'),
                      ('n1', 'c1'), ('n1', 'l1'), ('n1', 'o1'), ('o1', 'j1'), ('o1', 'b1'), ('o1', 'd1'),
                      ('a','x'),('f','x'),('f1','x'),('a1','x'),('b','x'),('b1','x')])


# In[369]:


trainingDatasetFinalUnseen = processingDataset(TariningDatasetUnseenDupRem)


# In[370]:


col2 = list(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
            'a1','b1','c1','d1','e1','f1','g1','h1','i1','j1','k1','l1','m1','n1','o1','x'])
concatDatasetUnseen = pd.DataFrame(trainingDatasetFinalUnseen, columns = col2)


# In[371]:


modelUnseen.fit(concatDatasetUnseen)


# In[372]:


# testing unseen


# In[373]:


testingDatasetFinalUnseen = np.array(processingDataset(TestingDatasetUnseenDupRem))


# In[374]:


testingFetureValuesUnseen =  np.transpose(np.transpose(testingDatasetFinalUnseen)[0:30])
print(testingFetureValuesUnseen.shape)
inferUnseen = VariableElimination(modelUnseen)


# In[ ]:


PredictedValuesUnseen = predictionFunction(testingFetureValuesUnseen,inferUnseen)


# In[ ]:


targetValueUnseen = np.transpose(np.transpose(testingFetureValuesUnseen)[30:31])
print("accuracy: " + accuracy(PredictedValuesUnseen,targetValueUnseen)

