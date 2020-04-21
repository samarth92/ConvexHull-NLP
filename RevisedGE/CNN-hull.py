
import numpy as np
import RevisedGE as re
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
import json
import sys

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation,TimeDistributed
from keras.layers import Bidirectional, LSTM
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate,BatchNormalization,MaxPooling1D, Convolution1D, Conv1D
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D, RepeatVector, Permute, merge
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate
from keras.callbacks import *
from keras.utils import to_categorical


import keras
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)
print(keras.backend.tensorflow_backend._get_available_gpus())


word_vectors = {}
EMBEDDING_DIM =50
f = open('glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = ''.join(values[:-EMBEDDING_DIM])
    coefs = np.asarray(values[-EMBEDDING_DIM:], dtype='float32')
    word_vectors[word] = coefs
f.close()

# ###################prepping text data#####################################
# texts=[]
# scores=[]
# lengths=[]

# f = open('./../reviews_Amazon_Instant_Video_5.json','r')
# for line in f:
#     data = json.loads(line)
#     texts.append(data['reviewText'])
#     scores.append(data['overall'])
#     lengths.append(len(data['reviewText'].split(' ')))
    
# df = pd.DataFrame()
# df['texts']=texts
# df['scores'] = scores
# df['lengths'] = lengths


# df=df[(df['lengths']>150) & (df['lengths']<300) ]
# # df=df.drop(columns=['lengths'])
# df_pos = df[df['scores']>=4]
# df_neg = df[df['scores']<=2]
# print(len(df_pos),len(df_neg))

# df_pos['target']=1
# df_neg['target']=-1

# df_pos = df_pos.sample(len(df_neg)*2,random_state=0)
# print(len(df_pos),len(df_neg))

# df = pd.concat([df_pos,df_neg]) 


# ################################


# class Args:
#     initial_size=-1
#     convexhull_size = -1
#     convergence_distance = 0.06
#     tolerated_qp_error = 0.03
#     convergence_change_rate = 0.01
#     sigma = 0.3

# args = Args()

# hulls=[]
# for index,row in df.iterrows():
#     sent=row['texts']
#     # sent = "A car can take you places Hundreds of researchers attempted to predict children’s and families’ outcomes, using 15 years of data. None were able to do so with meaningful accuracy Cars came into global use during the 20th century, and developed economies depend on them. The year 1886 is regarded as the birth year of the modern car when German inventor Karl Benz patented his Benz Patent-Motorwagen. Cars became widely available in the early 20th century. One of the first cars accessible to the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced animal-drawn carriages and carts, but took much longer to be accepted in Western Europe and other parts of the world."
#     vecs=[]
#     for word in sent.strip().split(' '):
#         try:
#             vecs.append(word_vectors[word.lower()])
#         except:
#             continue

            
#     nodes = np.asarray(vecs)
#     print(nodes.shape)
#     print(sent)
#     # if nodes.shape[0]<100: continue

#     args.initial_size = nodes.shape[0] // 10
#     args.convexhull_size = nodes.shape[0] // 5
#     re.args = args
#     convexhull_indexes = re.get_convexhull(nodes)
#     # print(convexhull_indexes)
#     hulls.append(convexhull_indexes)


# df['hulls']=hulls
# df.to_pickle('df.p')

df  = pd.read_pickle('df.p')
df = df.sample(frac=1, random_state=23).reset_index(drop=True)
df['target'][df['target']==-1]=0

hull_texts=[]
for index,row in df.iterrows():
    sent=row['texts'].strip().split(' ')
    hull_texts.append(' '.join([sent[i] for i in row['hulls']]))

df['hull_texts']=hull_texts
df_train, df_test = train_test_split(df, test_size=0.2,random_state=23)


print(sum(df_train['target']),len(df_train))
print(sum(df_test['target']),len(df_test))

####################################################################
def f1(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    # So we only measure F1 on the target y value:
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
#######################################################################################

############################training with full text####################################
MAX_SEQUENCE_LENGTH=70

tokenizer = Tokenizer(lower=True)
tokenizer.fit_on_texts(df_train['hull_texts'])
sequences = tokenizer.texts_to_sequences(df_train['hull_texts'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

train_X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
train_Y = df_train['target']
train_Y = to_categorical(np.asarray(train_Y))


sequences = tokenizer.texts_to_sequences(df_test['texts'])
test_X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
test_Y = df_test['target']
test_Y = to_categorical(np.asarray(test_Y))


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    try:
        word_vector = word_vectors[word]
#     if word_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = word_vector
    except KeyError:
        continue

#################
########################model definition######################################
# model = Sequential()

dropout_prob = [0.2,0.2]
hidden_dims = 50
filter_sizes  = (3,8)
num_filters = 10
BATCH_SIZE = 32

convs = []
filter_sizes = [3,4,5]

model=None
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

for fsz in filter_sizes:
    l_conv = Conv1D(nb_filter=128,filter_length=fsz,activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(5)(l_conv)
    convs.append(l_pool)
    
l_merge = merge.Concatenate(axis=1)(convs)
l_cov1= Conv1D(128, 5, activation='relu')(l_merge)
l_pool1 = MaxPooling1D(5)(l_cov1)
l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
l_pool2 = MaxPooling1D(3)(l_cov2)
l_flat = Flatten()(l_pool2)
l_dense = Dense(128, activation='relu')(l_flat)
preds = Dense(2, activation='softmax')(l_dense)

model = Model(sequence_input, preds)
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=["accuracy",f1])

print("model fitting - more complex convolutional neural network")
model.summary()
#############################################################################3

history = model.fit(train_X, train_Y, epochs=20,batch_size=BATCH_SIZE,validation_split=0.15,callbacks=[checkpointer])


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()

model.load_weights('weights.hdf5')
probs= model.predict(train_X,verbose=1,batch_size=100)
print("Train P R F Acc",metrics.precision_score(np.argmax(train_Y, axis=1), np.argmax(probs,axis=1)),
                        metrics.recall_score(np.argmax(train_Y, axis=1), np.argmax(probs,axis=1)),
                        metrics.f1_score(np.argmax(train_Y, axis=1), np.argmax(probs,axis=1)),metrics.accuracy_score(np.argmax(train_Y, axis=1), np.argmax(probs,axis=1)))
probs= model.predict(test_X,verbose=1,batch_size=100)
print("Test P R F Acc", metrics.precision_score(np.argmax(test_Y, axis=1), np.argmax(probs,axis=1)),
                        metrics.recall_score(np.argmax(test_Y, axis=1), np.argmax(probs,axis=1)),
                        metrics.f1_score(np.argmax(test_Y, axis=1), np.argmax(probs,axis=1)),metrics.accuracy_score(np.argmax(test_Y, axis=1), np.argmax(probs,axis=1)))
























# sent = "A car can take you places Hundreds of researchers attempted to predict children’s and families’ outcomes, using 15 years of data. None were able to do so with meaningful accuracy Cars came into global use during the 20th century, and developed economies depend on them. The year 1886 is regarded as the birth year of the modern car when German inventor Karl Benz patented his Benz Patent-Motorwagen. Cars became widely available in the early 20th century. One of the first cars accessible to the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced animal-drawn carriages and carts, but took much longer to be accepted in Western Europe and other parts of the world."
# vecs=[]
# for word in sent.strip().split(' '):
#     try:
#         vecs.append(word_vectors[word])
#     except:
#         continue

# fout = open('./data/temp.txt','w')
# fout.write(str(len(vecs))+'\t'+str(EMBEDDING_DIM)+'\n')
# i=0
# for vec in vecs:
#     fout.write(str(i)+'\t'+'\t'.join([str(j) for j in vec]))
#     fout.write('\n')
#     i+=1
# fout.close()