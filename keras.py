1. When you have a multi class classification problem and you are provided with lables that are ineteger encoded , you can directly use keras 
method to one hot encode them:

from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

-------------------------------------------------------------

2.
