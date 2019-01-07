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
