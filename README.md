# Intent_Classification
## Introduction
This repo was forked from https://github.com/Dark-Sied/Intent_Classification.
It is the source code of an intent classifier described in this medium article: https://towardsdatascience.com/a-brief-introduction-to-intent-classification-96fda6b1f557.     
It describes an intent classifier working with word embeddings and an LSTM Network (realized with Keras) capable of recognizing phrases and sentences of 21 different intents.

In this README file, I am going to document the code.

## Reading the Training Set
The training set for the Intent Classification is a csv-file in the following form:

```
<sentence 1>, <intent 1> CR
```

The file is read as a pandas dataframe:

```python
df = pd.read_csv("Dataset.csv", encoding = "latin1", names = ["Sentence", "Intent"])
```
This command creates an unique list of the intents:

```python
unique_intents = list(set(df["Intent"]))
```
## Word Cleaning

Word Cleaning removes all unnecessary characters from the sentences and transforms the sentences into a list of words. 

`sentences` is a list of sentences. `re.sub(r'[^ a-z A-Z 0-9]', " ", s)` replaces all characters different from `A-Z, a-z,` and `0-9` with blanks.    
`word_tokenize` splits each sentence into a list of words. Finally, upper case letters are transformed tolower case letters. The cleaned sentences are stored in `clean_sent` as a list of sentences, and each sentence is a list of words.

```python
import re
from nltk.tokenize import word_tokenize

clean_sent = []
for s in sentences:
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)
    w = word_tokenize(clean)
    clean_sent.append([i.lower() for i in w])
```

## Tokenizing

Tokenizing is representing every word (“token”) of our vocabulary as a vector, which can be processed by the learning model. 
We are using the tokenizer from TensorFlow, Keras. `clean_sent` is a list of sentences, every sentence is a list of words (output from Word Cleaning). filters specify the characters that are filtered out in the sentences (not necessary here as our words only contains the characters entences. `re.sub(r'[^ a-z A-Z 0-9]', " ", s)` replaces all characters different from `a-z` and `0-9`).

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

token = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
token.fit_on_texts(clean_sent)
VOCABULARY_SIZE = len(token.word_index) + 1
MAX_SENT_LENGTH = len(max(clean_sent, key = len))
encoded_sent    = token.texts_to_sequences(clean_sent)
padded_sent     = pad_sequences(encoded_sent, maxlen = MAX_SENT_LENGTH, padding = "post")
```

`fit_on_texts` creates a vocabulary of words (tokens) based on the input words. Each word is represented by an integer number.

`text_to_sequences(words)` transfers sentences (list of lists of words) into a list of lists of integer numbers. Each number is the representation (“token”) of a particular word of the vocabulary.

`pad_sequences(encoded, maxlen = max_length, padding = "post")` reads a list of lists of integer values (the tokens) and the maximum length of a sentence. It returns a list of lists of tokens, each sequence (sentence) has the length max_length. If a sentence is shorter than `max_length`, it is filled (“padded”) with 0 at the end.

## Tokenizing and Preparing the Intents (Targets)

`unique_intent` is a list of unique intents.

```python
out_token = Tokenizer(filters = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~')
out_token.fit_on_texts(unique_intent)
out_encoded = out_token.texts_to_sequences(intents)
```

`intents` is a `pandas.core.series` of intents. The length of the series is equal to the number of sentences.

`out_encoded` is a list of lists. Every list has one element: The token (integer value) of the intent of the corresponding sentence. 

```python
import numpy as np
np_encoded = np.array(out_encoded).reshape(len(encoded_output), 1)
```
This command transforms `out_encoded` into a numpy-array with the following shape:
```python
encoded_output.shape`: (1113, 1)
```
Now, this representation of the labels (intents) is transferred into a one-hot encoding:
```python
from sklearn.preprocessing import OneHotEncoder

one_hot = OneHotEncoder(sparse = False)
output_one_hot = one_hot.fit_transform(np_encoded)
```
output_one_hot is a 2-dimensional numpy.array with the following shape:
```python
output_one_hot.shape:
>> (1113, 21)
```
For every sentence, the intent of the sentence is represented with a one-hot vector of the size 21 (21 is the number of the different intents).

## Defining the Training and Validation Set

In this code, the data is split into a training set (80%) and a validation set (20%).
```python
from sklearn.model_selection import train_test_split

train_X, val_X, train_Y, val_Y = train_test_split(padded, output_one_hot, shuffle = True, test_size = 0.2)
```

Shape of train_X = (890, 28) and train_Y = (890, 21)
Shape of val_X = (223, 28) and val_Y = (223, 21)

890:	Number of sentences in the training set
223:	Number of sentences in the validation set
28:		Maximum length of a sentence
21: 	Number of intents and size of the one-hot output vector

## Defining and Compiling the Machine Learning Model

The neural network is a Sequential model form Keras. 
```python
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, 
                         Dropout

# vocab_size:  492
# max_length:  28

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length = max_length, 
          trainable = False))
model.add(Bidirectional(LSTM(128)))
model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(21, activation = "softmax"))
```
### Embedding-Layer

Changes an input integer value into a dense vector with 128 values. 

Bidirectional-LSTM Layer with 128 units

### Dense Layer 

Is a Feed-Forward Layer with 32 units

relu (Rectified Linear Unit) activation:
It returns element-wise max(x, 0)

### Dropout-Layer

rate = 0.5    
Dropout consists of randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

### Dense-Layer

with 21 output units for the 21 intents.
```python
model.compile(loss = "categorical_crossentropy", optimizer = "adam", 
              metrics = ["accuracy"])
model.summary()
```
This compiles the model and makes it ready for `training.model.summary()` 
shows a summary of the model.

## Training the Model

With the following code, the model is trained:
```python
from keras.callbacks import ModelCheckpoint
import time

filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss',  
                             verbose=1, save_best_only=True,
                             mode='min')

start = time.time()
hist = model.fit(train_X, train_Y, epochs = 100, batch_size = 32, 
                 validation_data = (val_X, val_Y),
                 callbacks = [checkpoint])
print("Elapsed time in seconds: ", time.time() - start)
```
The model has about 335k parameters. The weights-file model.h5 has a size of 3.5 MB.

## Testing the Model (Inference)

First, the weights have to be loaded into the model:
```python
model = load_model("model.h5")
```
Then a test sentence is treated in the same way as the training sentences:
First, all characters other than a-z, A-Z and 0-9 are replaced by space:
```python
clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
```
Then the sentence is separated into words and moved to lower case:
```python
test_words = word_tokenize(clean)
test_words = [w.lower() for w in test_words]
```
Now the words are transfered into integers (tokens):
```python
test_ls = word_tokenizer.texts_to_sequences(test_words)
```
Check for unknown words: Unknown words results in an empty list. If there are empty lists in test_ls, they are filtered out:
```python
if [] in test_ls:
    test_ls = list(filter(None, test_ls))
```
Then test_ls is reshaped into an 1-dimensional numpy-array.    
```python
test_ls = np.array(test_ls).reshape(1, len(test_ls))
```
At the end, the list of tokens is filled with zeros (“padded”) until `max_length` (the maximum length of a sentence):
```python
x = padding_doc(test_ls, max_length)
```
Finally, this np-array is fed into the model:
```python
pred = model.predict_proba(x)
```
pred is an array of probabilities for each label (intent).
