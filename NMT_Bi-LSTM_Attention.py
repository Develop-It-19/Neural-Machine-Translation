#Machine Translation can be used to map from one sequence to another.
  #Not only for Translating Human Languages but also for tasks like Date Format Translation.
#Attention Mechanism allows the network to focus on the Most relevant part of input while producing a specific part of output.
#Network using Attention Mechanism can translate from input of length Tx to output of length Ty.
  #Where Tx and Ty are of different lengths.
#You can Visualize where the network is paying Attention to while generating each output.

#Import Dependencies
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K

import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
%matplotlib inline

#Dataset
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

dataset[:10]

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)

index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])

#Define Shared Layer Objects as Global Variables.
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis = -1)
densor = Dense(1, activation = "relu")
activator = Activation(softmax, name = "attention_weights")
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
  s_prev = repeator(s_prev)
  concat = concatenator([a, s_prev])
  e = densor(concat)
  alphas = activator(e)
  context = dotor([alphas, a])
  
  return context

n_a = 64
n_s = 128
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation = softmax)

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
  X = Input(shape = (Tx, human_vocab_size))
  s0 = Input(shape = (n_s, ), name = 's0')
  c0 = Input(shape = (n_s, ), name = 'c0')
  s = s0
  c = c0

  outputs = []
  
  a = Bidirectional(LSTM(n_a, return_sequences = True))(X)
  
  for t in range(Ty):
    context = one_step_attention(a, s)
    
    s, _, c = post_activation_LSTM_cell(context, initial_state = [s, c])
    
    out = output_layer(s)
    
    outputs.append(out)
    
  model = Model(inputs = [X, s0, c0], outputs = outputs)
  
  return model

model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))

model.summary()

out = model.compile(optimizer = Adam(lr = 0.005, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01), loss = 'categorical_crossentropy', metrics = ['accuracy'])
out

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))

model.fit([Xoh, s0, c0], outputs, epochs = 10, batch_size = 100)

model.load_weights('models/model.h5')

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
  source = string_to_int(example, Tx, human_vocab)
  source = np.array(list(map(lambda x: to_categorical(x, num_classes = len(human_vocab)), source))).swapaxes(0, 1)
  prediction = model.predict([source, s0, c0])
  prediction = np.argmax(prediction, axis = -1)
  output = [inv_machine_vocab[int(i)] for i in prediction]
  
  print("source:", example)
  print("output:", ''.join(output))
  
#Print the Summary to find the Position of the Attention Layer.
model.summary()

#Plot the Attention Map to visualize where the network is paying attention while generating the Output.
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Friday October 28 1958, num = 6, n_s = 128)
                       

