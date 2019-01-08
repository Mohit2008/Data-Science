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

Visualisation:
  
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
