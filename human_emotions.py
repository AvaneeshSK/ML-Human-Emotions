# Human Emotions - Multi Classfication Problem
    # kaggle : https://www.kaggle.com/datasets/nelgiriyewithana/emotions


# Each entry in this dataset consists of a text segment representing a Twitter message and a corresponding label indicating the predominant emotion conveyed. The emotions are classified into six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5)


# Total Samples: 416809 
# |V| = 75302

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import tensorflow as tf
import tensorflow_hub as hub

import string
import random

from sklearn.metrics import confusion_matrix, accuracy_score

# FILE
df = pd.read_csv('Machine Learning 3/human_emotions.csv', low_memory=False)
df.rename(columns={'text':'Text', 'label':'Target'}, inplace=True)
df.drop(columns=['Unnamed: 0'], inplace=True)

# CHECK DISTRIBUTION OF LABELS
# print(df['Target'].value_counts())
# bars = plt.bar(['Joy', 'Sadness', 'Anger', 'Fear', 'Love', 'Surprise'], df['Target'].value_counts(), color=['blue', 'orange', 'green', 'red', 'purple', 'olive'])
# handles = []
# for bar in bars:
#     handles.append(Line2D([0], [0], color=bar.get_facecolor(), linewidth=5))
# plt.legend(handles=handles, labels=['Joy', 'Sadness', 'Anger', 'Fear', 'Love', 'Surprise'])
# plt.show()

# CHECK RANDOM 'SURPRISE' (type 5) LABEL
# surprise_texts = list(df[df['Target'] == 5]['Text'])
# print(len(surprise_texts))
# random.shuffle(surprise_texts)
# for i in range(10):
#     print(surprise_texts[i])
#     print('\n')

print(df)

# FEATURE, LABEL
texts = list(df['Text'])
targets = list(df['Target'])
chars = []
    # EXTRACT characters FROM TEXTS also : GET AVG, 95th PERCENTILE of texts, chars
    

word_lengths = []
char_lengths = []
for sentence in texts:
    string_ = ''
    word_lengths.append(len(sentence.split()))
    char_lengths.append(len(sentence))
    for c in sentence:
        string_ += f' {c} '
    chars.append(string_)
# print(np.percentile(word_lengths, 95)) # 41
# print(np.percentile(char_lengths, 95)) # 209
# print(np.mean(word_lengths)) # 19
# print(np.mean(char_lengths)) # 97

# GET TOTAL AMOUNT OF UNIQUE WORDS |V|
# unique_words = []
# for sentence in texts:
#     for word in sentence.split():
#         if word not in unique_words:
#             unique_words.append(word)
# print(len(unique_words)) # 75302


# SPLIT DATA
splitter = int(0.8 * len(texts)) # 333447

X_train_texts = texts[:splitter]
X_train_chars = chars[:splitter]
X_test_texts = texts[splitter:]
X_test_chars = chars[splitter:]
y_train = targets[:splitter]
y_test = targets[splitter:]


# TEXT PREPROCESSING
text_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    ngrams=1,
    max_tokens=75302,
    output_mode='int',
    output_sequence_length=41,
    pad_to_max_tokens=True
)
text_vectorizer.adapt(texts)
text_embedding = tf.keras.layers.Embedding(
    input_dim=75302,
    input_length=41,
    output_dim=128
)

char_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
    ngrams=1,
    max_tokens=len(string.ascii_lowercase) + 2, # +2 for <UNK>, whitespaces
    output_mode='int',
    output_sequence_length=209,
    pad_to_max_tokens=True
)
char_vectorizer.adapt(chars)
char_embedding = tf.keras.layers.Embedding(
    input_dim=len(string.ascii_lowercase) + 2,
    input_length=209,
    output_dim=25
)


# MODELLING
    # 1. USE MODEL (text data)
use_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'
use_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[], dtype='string'),
    hub.KerasLayer(use_url, trainable=False),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='softmax')
])
use_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
use_model.fit(
    x=np.array(X_train_texts),
    y=np.array(y_train),
    verbose=2,
    batch_size=128, 
    shuffle=True,
    epochs=10
)
preds_1 = use_model.predict(np.array(X_test_texts))
preds_1_max = []
for pred in preds_1:
    preds_1_max.append(np.argmax(pred))
use_model_score = accuracy_score(y_pred=preds_1_max, y_true=np.array(y_test))

    # 2. Conv1D Model (text data)
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, ), dtype='string'),
    text_vectorizer,
    text_embedding,
    tf.keras.layers.Conv1D(filters=64, activation='relu', kernel_size=5),
    tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
    tf.keras.layers.Conv1D(filters=64, activation='relu', kernel_size=5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='softmax'),
])
cnn_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
cnn_model.fit(
    x=np.array(X_train_texts),
    y=np.array(y_train),
    verbose=2,
    batch_size=128, 
    shuffle=True,
    epochs=10
)

preds_2 = cnn_model.predict(np.array(X_test_texts))
preds_2_max = []
for pred in preds_2:
    preds_2_max.append(np.argmax(pred))
cnn_model_score = accuracy_score(y_pred=preds_2_max, y_true=np.array(y_test))

# 3. RNN Bi LSTM MODEL (text data)
rnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, ), dtype='string'),
    char_vectorizer,
    char_embedding,
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=32, activation='tanh')),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='softmax')
])
rnn_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
rnn_model.fit(
    x=np.array(X_train_texts),
    y=np.array(y_train),
    verbose=2,
    batch_size=128, 
    shuffle=True,
    epochs=10
)

preds_3 = rnn_model.predict(np.array(X_test_texts))
preds_3_max = []
for pred in preds_3:
    preds_3_max.append(np.argmax(pred))
rnn_model_score = accuracy_score(y_pred=preds_3_max, y_true=np.array(y_test))


# 4. ANN Model (text data)
ann_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, ), dtype='string'),
    char_vectorizer,
    char_embedding,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=6, activation='softmax')
])
ann_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
ann_model.fit(
    x=np.array(X_train_texts),
    y=np.array(y_train),
    verbose=2,
    batch_size=128, 
    shuffle=True,
    epochs=10
)

preds_4 = ann_model.predict(np.array(X_test_texts))
preds_4_max = []
for pred in preds_4:
    preds_4_max.append(np.argmax(pred))
ann_model_score = accuracy_score(y_pred=preds_4_max, y_true=np.array(y_test))


# 5. Multi-Model - USE (text data), CNN 1D (text data), ANN (char data)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=[], dtype='string'),
    hub.KerasLayer(use_url, trainable=False),
    tf.keras.layers.Dense(units=32, activation='relu')
])

model_2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, ), dtype='string'),
    text_vectorizer,
    text_embedding,
    tf.keras.layers.Conv1D(filters=64, activation='relu', kernel_size=5),
    tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
    tf.keras.layers.Conv1D(filters=64, activation='relu', kernel_size=5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(units=32, activation='relu'),
])

model_3 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, ), dtype='string'),
    char_vectorizer,
    char_embedding,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64),
    tf.keras.layers.Dense(units=64),
    tf.keras.layers.Dense(units=32),
])

concat = tf.keras.layers.Concatenate()([model_1.output, model_2.output, model_3.output])
output_layer = tf.keras.layers.Dense(units=6, activation='softmax')(concat)
combined_model = tf.keras.Model(
    inputs=[model_1.input, model_2.input, model_3.input],
    outputs=output_layer
)

combined_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

combined_model.fit(
    x=[np.array(X_train_texts), np.array(X_train_texts), np.array(X_train_chars)],
    y=np.array(y_train),
    verbose=2,
    shuffle=True,
    batch_size=128, 
    epochs=10,
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2)]
)

preds_5 = combined_model.predict([np.array(X_test_texts), np.array(X_test_texts), np.array(X_test_chars)])
preds_5_max = []
for pred in preds_5:
    preds_5_max.append(np.argmax(pred))
combined_model_score = accuracy_score(y_pred=preds_5_max, y_true=np.array(y_test))

# CONFUSION MATRIX - Multi Model
sns.heatmap(data=confusion_matrix(y_pred=preds_5_max, y_true=y_test), fmt='.2f', annot=True, xticklabels=[0, 1, 2, 3, 4, 5], yticklabels=[0, 1, 2, 3, 4, 5])
plt.show()

# ALL SCORES PLOTTED

plt.style.use('ggplot')
bars = plt.bar(['USE Model', 'Conv1D Model', 'RNN Bi LSTM Model', 'ANN Model', 'Multi-Model'], [use_model_score, cnn_model_score, rnn_model_score, ann_model_score, combined_model_score], color=['lightcoral', 'burlywood', 'forestgreen', 'cornflowerblue', 'lightpink'])
handles = []
for bar in bars:
    handles.append(Line2D([0], [0], color=bar.get_facecolor(), linewidth=5))
plt.legend(handles=handles, labels=['USE Model', 'Conv1D Model', 'RNN Bi LSTM Model', 'ANN Model', 'Multi-Model'])
plt.show()
