
#loading required library

from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors

import re
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

import itertools
import datetime
import os

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge
import keras.backend as K
from keras.optimizers import Adadelta

#%%
# File paths

os.chdir("C:\\Users\\vamsi\\Desktop\\mis\\kerala")

TRAIN_CSV = "C:\\Users\\vamsi\\Desktop\\mis\\kerala\\Corpus.csv"
EMBEDDING_FILE = 'C:\\Users\\vamsi\\Desktop\\NLP\\google pretrainned embedings\\GoogleNews-vectors-negative300.bin.gz'

#%%
# Load training and test set
train_df = pd.read_csv(TRAIN_CSV)

stops = set(stopwords.words('english'))

#%%
#Understanding data 

type(train_df)

train_df.shape

train_df.isna()

train_df.count() #to the the coutnts for each variable

train_df.dropna().shape #to see no.of samples have non missing values

train_df.columns #to get the column names

train_df.head()

train_df.tail()

#%%
#droping missing data samples

train_df1 = train_df.dropna()

#%%
# rename the column names
train_df1.columns = ['s.no', 'text1']
train_df1.columns

#%%
#taking samole of trining data due to hardware issues
train_df1 = train_df1[1:100]

#%%
# making the same data copy train_df1 to train_df2 for cross joining 
train_df2 = train_df1.copy()

#%%
#Renaming the column names in copy train_df2
train_df2.columns = ['s.no', 'text2']
train_df2.columns

#%%
#cross join train_df1 with train_df2
train_df1["key"] = 1; train_df2["key"] = 1 ;
new_df = pd.merge(train_df1, train_df2, how='outer', on ="key"); del new_df["key"]

new_df.shape
new_df
#%%
# a copy of new_df for reference to add distance metrice t data set 
new_df_ref = new_df.copy()

#%%
#Adding manually label column "isduplicate" based on both the questions are same or not
new_df["isduplicate"] = np.where(new_df["text1"] == new_df["text2"], 1,0)

print(new_df["isduplicate"])

#%%

''' Pre process and convert texts to a list of words '''
#Normalization
#---------------
#Normalization generally refers to a series of related tasks meant to put all text on a level playing field: converting 
#all text to the same case (upper or lower), removing punctuation, expanding contractions, converting numbers to their
#word equivalents, and so on. Normalization puts all words on equal footing, and allows processing to proceed uniformly

def text_to_word_list(text):  
    text = str(text)
    text = text.lower()  
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    return text

#%%
# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding

#loading google pretrained word embedding
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

questions_cols = ['text1', 'text2']

#%%
# Iterate over  both training the questions of the dataset for replacing word with index num
for dataset in [new_df]:
    for index, row in dataset.iterrows():

        # Iterate through the text of both questions of the row
        for question in questions_cols:
 
            q2n = []  # q2n -> question numbers representation
            for word in text_to_word_list(row[question]):

                # Check for unwanted words
                if word in stops and word not in word2vec.vocab:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            # Replace "questions as word" to "question as number" representation
            dataset.set_value(index, question, q2n)


#%%         
embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

#%%
# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec


#%%
#checking max_seq_length for padding question with less length
max_seq_length = max(new_df.text1.map(lambda x: len(x)).max(),
                     new_df.text2.map(lambda x: len(x)).max())
                     
print(max_seq_length)


#%%
X = new_df[questions_cols]
Y = new_df['isduplicate']

# Split to dicts
X_train = {'left': X.text1, 'right': X.text2}

# Convert labels to their numpy representations
Y_train = Y.values

#%%
# Zero padding
for dataset, side in itertools.product([X_train], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

#%%
#Build the model

# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 25

#function for calculating manhattan_distance

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

#%%
# creating Input layer that specifies the shape of input data
    
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

#%%
#word embeddings via the Embedding layer for text data.

embedding_layer = Embedding(len(embeddings), embedding_dim,
                            weights=[embeddings], input_length=max_seq_length, 
                            trainable=False)    

#%%
# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

#lstm layes 50 hidden units
shared_lstm = LSTM(n_hidden)

#Output of laft and right swntences
left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

#%%
# Calculates the distance as defined by the maLSTM model
malstm_distance = Merge(mode=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                        output_shape=lambda x: (x[0][0], 1))([left_output, right_output])

#%%
# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

#%%
# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

#%% 
#loss compilation
malstm.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

#%%
# Start training
training_start_time = time()

malstm_trained = malstm.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, nb_epoch=n_epoch)
                            

print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

#%%
predicted = malstm.predict([X_train['left'], X_train['right']])

#%%
print(predicted)
print(predicted.shape)

#%%
new_df_ref['manhattan_distance'] = predicted
print(new_df_ref)

#%% 
#Summary of the model
print(malstm.summary())

#%%
print(malstm_trained.params)

#%%
for weight in malstm.get_weights(): # weights from Dense layer omitted
    print(weight.shape)

#%%
for weight in malstm.get_weights(): # weights from Dense layer omitted
    print(weight)     


#%%
#Plotting the results
# Plot accuracy
plt.plot(malstm_trained.history['acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()
# Plot loss
plt.plot(malstm_trained.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()




#https://aws.amazon.com/blogs/machine-learning/build-a-social-media-dashboard-using-machine-learning-and-bi-services/

#https://aws.amazon.com/blogs/machine-learning/detect-sentiment-from-customer-reviews-using-amazon-comprehend/

#https://unamo.com/en/social-media-monitoring

#https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html

#https://www.kdnuggets.com/2017/02/natural-language-processing-key-terms-explained.html

#example : https://datascience.stackexchange.com/questions/11402/preprocessing-text-before-use-rnn/11421

#https://gallery.azure.ai/Experiment/Text-Classification-Step-2-of-5-text-preprocessing-2

#https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/

#http://www.nltk.org/book/ch03.html

