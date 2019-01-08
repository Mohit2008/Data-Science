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
  
  model.save('cats_and_dogs_small_1.h5')
  
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
  
  
  
  
  
  
  
  
