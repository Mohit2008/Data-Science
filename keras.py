1. When you have a multi class classification problem and you are provided with lables that are ineteger encoded , you can directly use keras 
method to one hot encode them:

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

-------------------------------------------------------------

2. A simple model in keras looks like the following:
  
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,))) # you can evel pass the regulariser or an weight initialiser
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', # could be adam or momentum 
                loss='categorical_crossentropy', # could be mse
                metrics=['accuracy']) # could be mae
network.fit(train_images, train_labels, epochs=5, batch_size=128) # can also pass the validation_data

-------------------------------------------------------------

3. While working with image data always remember to vectorise them and scale them by dividng by 255, and for text data always ensure that there 
sequence length is same for every input to network , keras has a way to pad the swequences to a given length.

-------------------------------------------------------------

4. setting the loss and metric can be done using below:

from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

-------------------------------------------------------------
                                                                                
5. Run the validation logic, which will test the performance of model on validation data after every epoch and list down the metric on validation and training data :
  
 x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
-------------------------------------------------------------

6. To visualize what your network has gone through on training and validation data there is a history object returned by the model.fit method
which contain information about the loss and metric on training and validation data , plotting it will give you an insight as to when did the 
overfitting happened or can be used accross different models to compare there performance:
  
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
history_dict = history.history
history_dict.keys()
# dict_keys(['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy'])

* Visualisation:
  
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss') # same could be done for accuracy
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss') # same could be done for accuracy
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

-------------------------------------------------------------

7. Two ways of dealing with text data is to vectorise them by representing every sample by a one hot encoded vector(naive approach) or to 
train an Embedding which will represent every word with a vector that encoded is latent meaning. You can even use a pretrained embedding when you dont 
have sufficiently large data , the input to such layer in the integer encoded sequence of words that belong to a sample , you will then have 
to pad them to ensure they are of same length and finally pass them to a Embedding layer.

-------------------------------------------------------------

8. When you have multiple lables , you have two options , one is you encode them using one hot encoding with categorical_crossentropy as loss:
  
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

Or you can use them as integer but the loss function you would have to use would be sparse_categorical_crossentropy

-------------------------------------------------------------

9. For regression you should not use any activation function in the output layer and the loss function that you should try on is mse:
  
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    
 -------------------------------------------------------------

10. The following strategies could be useful to prevent the model from overfitting:
  - Get more data(most important)
  -Adjust your network architecture, bigger network give more room to learn the patterns leading to overfitting of your model, a smaller 
  network can act as a bottleneck not letting network learn enough patterns causing underfitting.
  - Add requilarisation contraints such as l1,l2, droput, batch normalisation, wieghts initialiser.
  
  
  The general workflow to find an appropriate model size is to start with relatively few layers and parameters, and start increasing 
  the size of the layers or adding new layers until you see diminishing returns with regard to the validation loss.
  
* Weights regulariser:
    
from keras import regularizers

l2_model = models.Sequential()
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu', input_shape=(10000,))) # regularizers.l1_l2(l1=0.001, l2=0.001), regularizers.l1(0.001)
l2_model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                          activation='relu'))
l2_model.add(layers.Dense(1, activation='sigmoid'))


* Droput:
  
dpt_model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) # input_shape tells the shape of one sample along with comma where you can have any sample size
dpt_model.add(layers.Dropout(0.5))

-------------------------------------------------------------
  
11. Convnets:
    
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # you have 32 filters of size 3*3 and stride 1, notice the input shape this contains the dimensiosn of one sample
model.add(layers.MaxPooling2D((2, 2))) # you have a max pooling of size 2*2 and stride 2
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten()) # flatten out the tensor into one vector of length 64*3*3
model.add(layers.Dense(64, activation='relu')) # create fully connected layer
model.add(layers.Dense(10, activation='softmax')) # apply oputput layer and softmax function

model.summary() # see how the model architecture looks like 
  
-------------------------------------------------------------
  
12. Always remember to vectorise the image in float32 and divide by 255 

train_images = train_images.reshape((60000, 28, 28, 1)) # depending upon no of channels
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
  
 -------------------------------------------------------------

13. To create an iterator of your directory to pull in the images data use the following:
  
from keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,target_size=(150, 150),batch_size=20,class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,target_size=(150, 150), batch_size=20,class_mode='binary')
 _____________
  
If you have data loaded as numpy array:
  
train_generator=train_datagen.flow(training_images, 
                                   training_labels, 
                                   batch_size=512)

validation_generator = validation_datagen.flow(
    testing_images,
    testing_labels,
    batch_size=64
)

* When every you are using some generator function you will now have to use fir_generator, evaluate_generator, predict_generator which 
is provided by keras which lets it know that the data has to be generated from a generator function which in our case is ImageDataGenerator:
  
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100, # the generator will run infinetly and thus we need to indicate to keras how many round of generation will include all our data or complete one epoch, we have 2000 images in train and batch size was 20 so 20*100=2000
      epochs=30,
      validation_data=validation_generator, 
      validation_steps=50) # same for validation data

-------------------------------------------------------------
  
14. You can save your model in keras to disk:
  from keras.models import load_model

  model.save('cats_and_dogs_small_1.h5')
  model = load_model('cats_and_dogs_small_2.h5')
  
-------------------------------------------------------------
  
 15. Overfitting in Convnets can be significantly reduces by data augumentation and droput , you can augument the data using the fowlloing:
  
  train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40, #s a value in degrees (0-180), a range within which to randomly rotate pictures.
    width_shift_range=0.2, # are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
    height_shift_range=0.2, # are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically or horizontally.
    shear_range=0.2, # is for randomly applying shearing transformations.
    zoom_range=0.2, # s for randomly zooming inside pictures.
    horizontal_flip=True,# is for randomly flipping half of the images horizontally 
   fill_mode='nearest',) # is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
  
  
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,target_size=(150, 150), batch_size=32, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,target_size=(150, 150),batch_size=32,class_mode='binary')

history = model.fit_generator(
      train_generator,steps_per_epoch=100,epochs=100,validation_data=validation_generator,validation_steps=50)

-------------------------------------------------------------
  
  16. Pretrained Convnet
  
  Uisng the pretrained network you can perform 2 things one is to just use the conv base of pre trained network and use that to extract new features
  which will be fed into a classifier that you will train from scratch or you can perform fine tuning where you fix the first few layers of previosuly
  trained convnet and will update the last few layers along with dense layers to build your classifier
  
  
  
from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', # to specify which weight checkpoint to initialize the model from
                  include_top=False, # not include dense layer
                  input_shape=(150, 150, 3)) # the shape of image that we are going to pass to network

* You have two options from here , one is that you can directly pass in all your images and run through the predict method of this conv_base and
save it to disk or numpy array in which case your images have to pass through the conv_base only once and this will be quick however you wont be 
able to augument your images this was.

The second option is to create a netwrok with the conv_base and desnly connected layer and train the network by setting the conv_base trainiable param to 
false and pass in the data through entire netwrok this is slow but will make use of data augumentation.
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=30,
      validation_data=validation_generator,
      validation_steps=50,
      verbose=2)

* In FineTuning process you are going to set the training for last few layers of the conv_base For the same reason, it is only possible 
to fine-tune the top layers of the convolutional base once the classifier on top has already been trained. If the classified wasn't 
already trained, then the error signal propagating through the network during training would be too large, and the representations 
previously learned by the layers being fine-tuned would be destroyed. 
Thus the steps for fine-tuning a network are as follow:

1) Add your custom network on top of an already trained base network.
2) Freeze the base network.
3) Train the part you added.
4) Unfreeze some layers in the base network.
5) Jointly train both these layers and the part you added.:
 
We have already completed the first 3 steps when doing feature extraction. Let's proceed with the 4th step: 
we will unfreeze our conv_base, and then freeze individual layers inside of it.

conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
  
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50)
  
-------------------------------------------------------------
  
  17. A few remarkable things to note here:

- The first layer acts as a collection of various edge detectors. At that stage, the activations are still retaining almost all of the 
information present in the initial picture.
- As we go higher-up, the activations become increasingly abstract and less visually interpretable. They start encoding higher-level 
concepts such as "cat ear" or "cat eye". Higher-up presentations carry increasingly less information about the visual contents of 
the image, and increasingly more information related to the class of the image.
- The sparsity of the activations is increasing with the depth of the layer: in the first layer, all filters are activated by the 
input image, but in the following layers more and more filters are blank. This means that the pattern encoded by the filter isn't found in the input image.
 
-------------------------------------------------------------
  
18. Text processing, converting text to sequence or one hot encoded arrays:
  
from keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words=10000,oov_token="<NEW>") # use num_words to have which would be used in text_to_seq method
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples,padding="post", truncating="post",  maxlen=<>) # integer sequence
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary') # one hot encoded
word_index = tokenizer.word_index # get the token dictionary



-------------------------------------------------------------
  
19. Pad the sequence such that each sample is of same length:
    
from keras import preprocessing
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen) # pad the integer sequence with some length =maxlen, larger sequence are dropped smaller one are padded with 0
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

-------------------------------------------------------------
 
20. Use Embedding layer:
  
from keras.layers import Embedding

# The Embedding layer takes at least two arguments:
# the number of possible tokens, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 64.
embedding_layer = Embedding(1000, 64,input_length=<seq_length>) # for 1000 tokens represent each token with a 64 dim vector

Notebook - https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/Course%203%20-%20Week%202%20-%20Lesson%201.ipynb#scrollTo=b5S6-58FPtvL
         - (Text Classification Multi class) -https://colab.research.google.com/drive/1_6JuE7gDiPG33JgyIYCBmdp_8u0tl9cs
         -(Pretrained Embedding+LSTM) - https://colab.research.google.com/drive/1nrxfefixKSB_LvldhfDoD0kEift6cQ8n
          
      
-------------------------------------------------------------
  
21. Use a pre trained embedding:
    
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix]) # set the pre trained embedding in the first layer
model.layers[0].trainable = False # set the training of this to be false

-------------------------------------------------------------
  
  
22. For using custom embedding below is the kind of network:
  
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen)) # will output a 3d tensor (samples, maxlen, embedding_dim)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))


You can imagine the output from a embedding to be a tensor of 3d where each sample will be of 2 dim like image but in case of text it would be 
no of words, embedding_dim. For ex if you have 5 words in a sequence then your single sample will be represneted by (5,embedding_dim).

* When you pass the above output into a RNN it will process this sequence one at a time such that at every timestep it takes one sequence 
and use that to calculate output which is then fed as a input to next sequence and this happens for the amaount of sequence length.
-------------------------------------------------------------
  
24. SimpleRNN
  
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True)) # when you want to get the ouput from every sequence processed such that it feds into the next rnn layer. It gives tensor of size (batch_size, timesteps, output_features)
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # This last layer only returns the last outputs.
model.summary()
  
  Simple Rnn suffers from the problem of vanishing gradient and cannot learn long term temporal dependency and hence are not that useful
-------------------------------------------------------------
    
25. LSTM
   
  
 The lstm consist of a long term memory and short term memory along with multiple gates like use gate ,forget gate , remember gate etc which 
continously keep updating the long term and short term memory which enable it to capture long term dependency 
 from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32)) # since we have a single layer we didnt had return sequence
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
    
-------------------------------------------------------------

26. When you define an LSTM,RNN,GRU layer it takes a param which is the no of units , so let say that the number of unit=32 which means that 
  you are going to have 32 neurons in that layer and  what will get output from that layer will have one of the dimersion equal to 32 for example in a
  simple rnn wirth just one sample you will have input of shape (no_of_timesteps/no_of_sequences, input_features/embedding vector) , the 
  state matric that gets computed and used at every timestep will be of shape (32/no_of_units_output_features), weight matrix W will be of shape
  (output_features/32, input_features) , U which is state weights is of size (output_features/32, output_features/32) and bias b is of shape
  (output_features/32). In a simple RNN the computation will be like output_t= activation(W*input_t+U.state_t+b0).
  
  In general embedding matrix gived 3d tensor of size(batch_size, timesteps, features) which gets fed in LSTM/GRU/RNN and generates output of shape
  (batch_size,timesteps ,num_units) in case return_sequence or  (batch_size ,num_units) in normal case. Generally from the last layer of these 
  recurrent cells we get ouput of shape (batch_size ,num_units) which does not require any flattening for upstream dense layer.
                                                                
  
-------------------------------------------------------------
  
27. Three advance technique to increase the performance of your netwrok is :
    
  -Recurrent dropout, a specific, built-in way to use dropout to fight overfitting in recurrent layers.
  -Stacking recurrent layers, to increase the representational power of the network (at the cost of higher computational loads).
  -Bidirectional recurrent layers, which presents the same information to a recurrent network in different ways, increasing accuracy and mitigating forgetting issues.
  
-------------------------------------------------------------
   
28. To create a generator for a time series data....refer to the below notebook:
    
  https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/6.3-advanced-usage-of-recurrent-neural-networks.ipynb
  
  
-------------------------------------------------------------
    
29. Droput regularisation in GRU/LSTM includes a dropout and recurrent_droput:
  
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, dropout=0.2,recurrent_dropout=0.2,input_shape=(None, float_data.shape[-1]))) # notice the 2 droputs that we used here
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)
 
-------------------------------------------------------------
   
30. Other way of comabting overfitting in LSTM/GRU is by increasing the capacity of your netwrok by stacking more layers:
  
  model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1, 
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

-------------------------------------------------------------
  
31. The third technique that can reduce overfitting in some cases is to use bidirectional recurrent meural network, we know that rnn
  c\are capable of extracting patterns when we have order dependent or time dependent information. NLP domain is more order dependent and 
  time series is more time dependent.
  
    A bidirectional RNN exploits the order-sensitivity of RNNs: it simply consists of two regular RNNs, such as the GRU or LSTM layers 
    that you are already familiar with, each processing input sequence in one direction (chronologically and antichronologically), 
    then merging their representations. By processing a sequence both way, a bidirectional RNN is able to catch patterns that may have 
    been overlooked by a one-direction RNN.
    
    Incase of time series you have temporal dependncy where processing chronological order makes more sense rather than rever order since recent past
    makes more sense for forecasting future information. Thus bidirectional rnn are not very great in processing time series but can show good 
    performance for other task specifically in NLP.
    
    Thus, remarkably, on such a text dataset, reversed-order processing works just as well as chronological processing, confirming our hypothesis 
    that, albeit word order does matter in understanding language, which order you use isn't crucial. Importantly, a RNN trained on 
    reversed sequences will learn different representations than one trained on the original sequences.
    
from keras import backend as K
K.clear_session()
model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32))) # pass in the lstm instance, a new instance to process the data in reverse order will be automatically created
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)    
    
    
-------------------------------------------------------------
    
32. Sequence data can also be processed in a way faster way using 1d convolution such that the conv layer extracts feature whereever it exsist in 
the sequence.

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))
    

-------------------------------------------------------------

33. You can even combine the convolution with recurrent layers so that the conv layers extract features from the sequence and then use 
feature map to lears the states between the feature maps 

model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(layers.Dense(1))

------------------------------------------------------------

34. The sequential model makes the assumption that the network you want to create has one input and one output and it consist of a linear stack 
of layers. This setup worked for most of the network but for more advanced setup we might have n/w taking multiple inputs, branching out , looking
more like a graph.You can have networks with multiple inputs , imagine a hybrid network you have a numerical data that you want to pass directly
to dense layer, you also have text data that you would like to pass through LSTM and you also have image data that you would like to pass through
the convolution layer , now you can do this with a sequention layer since you now have 3 inputs so you would need to make use of functiona 
api of keras. Also you can have a n/w with multiple output given a input , or you could have netwroks like inception and residual in which case
you need function api

from keras.models import Model
from keras import Input
from keras import layers

input_tensor = Input(shape=(64,))
x= layers.Dense(32, activation="relu") (input_tensor)
x= layers.Dense(32, activation="relu") (x)
output_tensor= layers.Dense(10, activation="softmax") (x)
model=Model(input_tensor,output_tensor)
model.summary()
model.compile(.....)
model.fit(.....)

------------------------------------------------------------

35. To speed up the computation of the Convnets there are different varaition of convnets that you can consider:
  
  -Dilated Kernels- A 3x3 kernel with a dilation rate of 2 will have the same field of view as a 5x5 kernel, while only using 9 parameters.
                    Imagine taking a 5x5 kernel and deleting every second column and row.
  -Spatial Seperable Kernel- A spatial separable convolution simply divides a kernel into two, smaller kernels. The most common case would 
                              be to divide a 3x3 kernel into a 3x1 and 1x3 kernel. The main issue with the spatial separable convolution is 
                              that not all kernels can be “separated” into two, smaller kernels.
  -Depthwise Separable Convolutions- This involve the depthwise convolution and the pointwise convolution. We do so by using 3 kernels of shape 5x5x1
                                     Each 5x5x1 kernel iterates 1 channel of the image (note: 1 channel, not all channels), getting the scalar products 
                                      of every 25 pixel group, giving out a 8x8x1 image. Stacking these images together creates a 8x8x3 image.
                                      Therefore, we iterate a 1x1x3 kernel through our 8x8x3 image, to get a 8x8x1 image. We can create 256 1x1x3 
                                      kernels that output a 8x8x1 image each to get a final image of shape 8x8x256. 12x12x3 — (5x5x1x3) — > (1x1x3x256) — >12x12x256.
                                      the main difference is this: in the normal convolution, we are transforming the image 256 times. And every transformation uses up 
                                      5x5x3x8x8=4800 multiplications. In the separable convolution, we only really transform the image once — in the depthwise convolution.
                                      Then, we take the transformed image and simply elongate it to 256 channels. Without having to transform the image over and over again, 
                                       we can save up on computational power.
                                      https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
                                      https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
layers.SeperableConv2D(32, (3,3), activation="relu", padding="same"......)

------------------------------------------------------------

36. Btach normalisation ensures that your data gets normalised as it passes through the layers , can be used with cnn, dense, rnn

from keras.layers.normalization import BatchNormalization

# instantiate model
model = Sequential()

# we can think of this chunk as the input layer
model.add(Dense(64, input_dim=14, init='uniform'))
model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dropout(0.5))


------------------------------------------------------------

37. Hyperas, HyperOpt are great tools for performing hyperparameter optimisation and integrates with keras. Its good to also use enseble
techniques with DL model, one combination that worked well was combining (randomForest/GradinetBoosting, deep neural n/w)

------------------------------------------------------------

38. You can speed up the model generation time and the understanding by implementing callbacks function which are functions that are going 
to perform certain task while the model is running for example doing spmething after every epoch, or after evebry batch. Keras provides various
callback which could be very useful:
  
* BaseLogger-Callback that accumulates epoch averages of metrics.
* ModelCheckpoint- Save the model after every epoch. keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
* EarlyStopping - Stop training when a monitored quantity has stopped improving. keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
* ReduceLROnPlateau- Reduce learning rate when a metric has stopped improving.
* TensorBoard -This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics, 
                as well as activation histograms for the different layers in your model.
  
In the call to mode.fit you pass the callbacks=[] list of callback functions that you have defined

------------------------------------------------------------

39. To visualize the activations generated from each of the conv layer of your model you can use the below code:
  
 Here we have a conv model which has following layers:
  
 [<tensorflow.python.keras.layers.convolutional.Conv2D at 0x7f95808af668>,
 <tensorflow.python.keras.layers.pooling.MaxPooling2D at 0x7f95208b3080>,
 <tensorflow.python.keras.layers.convolutional.Conv2D at 0x7f95208b3630>,
 <tensorflow.python.keras.layers.pooling.MaxPooling2D at 0x7f95208b3a20>,
 <tensorflow.python.keras.layers.core.Flatten at 0x7f95208b3d30>,
 <tensorflow.python.keras.layers.core.Dense at 0x7f95208480f0>,
 <tensorflow.python.keras.layers.core.Dense at 0x7f9520848630>]
  
import matplotlib.pyplot as plt

FIRST_IMAGE=3
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 4 # which filter you want to visualize
plt.imshow(test_images[FIRST_IMAGE].reshape(28,28))
plt.show()
plt.imshow(test_images[SECOND_IMAGE].reshape(28,28))
plt.show()
plt.imshow(test_images[THIRD_IMAGE].reshape(28,28))
plt.show()
f, axarr = plt.subplots(3,4)
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers] # get the ouput from all the 7 layers 
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  # Visualize the first 4 layers which contains the conv and max pooling layers
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)

 ------------------------------------------------------------

40. Use the history object to visualise the model performance:
  
acc      = history.history[     'acc' ]
val_acc  = history.history[ 'val_acc' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )


41. You can have the concept of shared layers using keras functional api , you can either share layers or can share a model

# Load the input layer from keras.layers
from keras.layers import Input

n_teams = unique(games_season['team_1']).shape[0]
team_lookup = Embedding(input_dim=n_teams,
                        output_dim=1,
                        input_length=1,
                        name='Team-Strength')

teamid_in = Input(shape=(1,))
strength_lookup = team_lookup(teamid_in) # Lookup the input in the team strength embedding layer
strength_lookup_flat = Flatten()(strength_lookup) # Flatten the output
team_strength_model = Model(teamid_in, strength_lookup_flat, name='Team-Strength-Model') # Combine the operations into a single, re-usable model

team_in_1 = Input((1,),name="Team-1-In") # Input layer for team 1
team_in_2 = Input((1,), name="Team-2-In")# Separate input layer for team 2

team_1_strength = team_strength_model(team_in_1) # Lookup team 1 in the team strength model
team_2_strength = team_strength_model(team_in_2) # Lookup team 2 in the team strength model


You can combine layers in kears in multiple ways (add, subtract, multiply, concatenate). All layers have to have 
same shape for add , subtract and multiply to work whereas you can use concatenate when you have layers of different size

from keras.layers import Subtract

# Create a subtract layer using the inputs from the previous exercise
score_diff = Subtract()([team_1_strength, team_2_strength])
model = Model([team_in_1, team_in_2], score_diff)

# Compile the model
model.compile('adam', loss='mean_absolute_error')

with such a model with multiple inputs you need to supply a list of inputs when callsing the fit , predict, evaluate methods

input_1 = games_season['team_1']
input_2 = games_season['team_2']

# Fit the model to input 1 and 2, using score diff as a target
model.fit([input_1, input_2],
          games_season['score_diff'],
          epochs=1,
          batch_size=2048,
          validation_split=0.1,
          verbose=True)


42. You can create a model with multiple outputs 

input_tensor = Input((2,))
output_tensor_1 = Dense(1, activation='linear', use_bias=False)(input_tensor) # Create the first output
output_tensor_2 = Dense(1, activation='sigmoid', use_bias=False)(input_tensor) # Create the second output 
model = Model(input_tensor, [output_tensor_1, output_tensor_2]) # Create a model with 2 outputs
model.compile(loss=['mean_absolute_error', 'binary_crossentropy'], optimizer=Adam(lr=0.01))


43. Notebooks

1. Image Generator + Visualizing the Conv layers for a input and see all the activations/feature maps at once:
  https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%208%20-%20Lesson%203%20-%20Notebook.ipynb
  https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb
    
2. Custom callbacks:
  https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%204%20-%20Lesson%204%20-%20Notebook.ipynb
   
3. Image Data Augumentation + Transfer Learning

(Binary class) https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb#scrollTo=BMXb913pbvFg
(Multi-Class) https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%208%20-%20Lesson%202%20-%20Notebook%20(RockPaperScissors).ipynb#scrollTo=it1c0jCiNCIM
    
    
44. Create dir structure for keras:

shutil.rmtree('/tmp')
try:
    directory="/tmp/covid_vs_normal"
    if not os.path.exists(directory):
        os.makedirs(directory+"/training/covid")
        os.makedirs(directory+"/training/normal")
        os.makedirs(directory+"/testing/covid")
        os.makedirs(directory+"/testing/normal")
except OSError:
    pass

# modify this logic based upon the need , here i had lot many examples in normal class and only 
# 200 examples in covid class so i had to do downsampling of normal hence the additional logic 
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE, SPLIT):
    if SPLIT=="NORMAL": # to downsample examples for this class
      all_files=os.listdir(SOURCE)
      down_Sample_files=random.sample(all_files, int(len(all_files)*0.17)) # get only 200 examples 
      train_files=random.sample(down_Sample_files, int(len(down_Sample_files)*SPLIT_SIZE))
    else:
      all_files=os.listdir(SOURCE)
      train_files=random.sample(all_files, int(len(all_files)*SPLIT_SIZE))
    #print(train_files)
    for f in train_files: copyfile(SOURCE+f, TRAINING+f) # set up the train directory
    for f in os.listdir(SOURCE): # set up the lest dir
      if SPLIT=="NORMAL": 
        if f not in train_files and f in down_Sample_files:
            copyfile(SOURCE+f, TESTING+f)
      else:
        if f not in train_files:
            copyfile(SOURCE+f, TESTING+f)


COVID_SOURCE_DIR = "/content/COVID-19 Radiography Database/COVID-19/
TRAINING_COVID_DIR = "/tmp/covid_vs_normal/training/covid/"
TESTING_COVID_DIR = "/tmp/covid_vs_normal/testing/covid/"
NORMAL_SOURCE_DIR = "/content/COVID-19 Radiography Database/NORMAL/
TRAINING_NORMAL_DIR = "/tmp/covid_vs_normal/training/normal/"
TESTING_NORMAL_DIR = "/tmp/covid_vs_normal/testing/normal/"

split_size = .9
split_data(COVID_SOURCE_DIR, TRAINING_COVID_DIR, TESTING_COVID_DIR, split_size, "COVID") # gen dir to covide
split_data(NORMAL_SOURCE_DIR, TRAINING_NORMAL_DIR, TESTING_NORMAL_DIR, split_size, 'NORMAL') # gen dir for normal


45. Kaglle utility for downaloding data 

! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download --unzip '<username>/<datasetname>'

46. Loading a pre trained embediing and pass it to keras

!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt \
    -O /tmp/glove.6B.100d.txt
embeddings_index = {};
with open('/tmp/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;
        
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    # YOUR CODE HERE - experiment with combining different types, such as convolutions and LSTMs
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, activation="relu", return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation="relu")),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

num_epochs = 50
history = model.fit(training_sequences, training_labels, epochs=num_epochs, validation_data=(test_sequences, test_labels), verbose=2)
