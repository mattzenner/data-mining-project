# https://autokeras.com/tutorial/structured_data_classification/
# https://www.kaggle.com/madz2000/cnn-using-keras-100-accuracy

# This is a reproduction of the kaggle above. Basically just copied the code in
# the notebook provided and made small adjustments to values to customize

import keras # api using TensorFlow as backend; makes TF more abstract, easier
import pickle # save and load python objects
import numpy as np # NUMBERS
import pandas as pd # DATA
import tensorflow as tf # google
import autokeras as ak
import seaborn as sns # nice plots
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau

num_classes = 29
test_ratio = 1./100. * 10.0
train_file_path = "D://archive//test_A.csv" # filepaths to training and testing data, test_A not the best name
test_file_path = "D://aslmnist//sign_mnist_test//sign_mnist_test.csv"

train_df = pd.read_csv(train_file_path) # read csv to dataframes
# test_df = pd.read_csv(test_file_path)

# test = pd.read_csv(test_file_path)
# y = test['label'] # y is labels, output of the model(which letter is being signed?)

print(train_df.head()) # print the first few lines

plt.figure(figsize = (10, 10)) # show a plot counting the amounts of training images in each class, each class should have about the same amount
sns.set_style('darkgrid')
sns.countplot(train_df['label'])
plt.show()

# (x_train, x_test), (y_train, y_test) = temp
train_df, test_df = train_test_split(train_df, test_size=test_ratio)

y = test_df['label'] # y is labels, output of the model(which letter is being signed?)
y_train = train_df['label'] # create y
y_test = test_df['label']
del train_df['label']
del test_df['label']

from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

x_train = train_df.values
x_test = test_df.values

# Normalize the data
x_train = x_train / 255
x_test = x_test / 255

# Reshaping the data from 1-D to 3-D as required through input by CNN's
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

f, ax = plt.subplots(2,5)
f.set_size_inches(10, 10)
k = 0
for i in range(2):
    for j in range(5):
        ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
        k += 1
    plt.tight_layout()

plt.show()


# With data augmentation to prevent overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

model = Sequential() # initializing the model and appending layers to it
model.add(Conv2D(75, (3, 3), strides=1, padding='same', activation='relu', input_shape=(28, 28, 1))) # convolutional layer
model.add(BatchNormalization()) # normalization layer
model.add(MaxPool2D((2, 2), strides=2, padding='same')) # pooling layer
model.add(Conv2D(50, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2)) # dropout layer
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides = 2, padding='same'))
model.add(Conv2D(25, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units = 512, activation='relu')) # dense layer used towards end of model
model.add(Dropout(0.3))
model.add(Dense(units = num_classes, activation='softmax')) # final layer, has 29 units, one for each possible class label
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() # print summary

history = model.fit(datagen.flow(x_train, y_train, batch_size=128), epochs=20, validation_data=(x_test, y_test), callbacks=[learning_rate_reduction])

print("Accuracy of the model is - ", model.evaluate(x_test, y_test)[1]*100 , "%")

epochs = [i for i in range(20)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(16,9)

ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy') # make a nice plot
ax[0].plot(epochs, val_acc, 'ro-', label='Testing Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, 'g-o', label='Training Loss')
ax[1].plot(epochs, val_loss, 'r-o', label='Testing Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()

predictions = np.argmax(model.predict(x_test), axis=-1) # model.predict_classes(x_test)
for i in range(len(predictions)):
    if(predictions[i] >= 9):
        predictions[i] += 1
predictions[:5]

classes = ["Class " + str(i) for i in range(1, num_classes + 1)]
print("Size of target_names: ", len(classes))
print(classification_report(y, predictions, target_names=classes))

cm = confusion_matrix(y, predictions)

cm = pd.DataFrame(cm , index=[i for i in range(num_classes + 1)], columns=[i for i in range(num_classes + 1)])

plt.figure(figsize = (15, 15))
sns.heatmap(cm, cmap="Blues", linecolor='black', linewidth=1, annot=True, fmt='')

correct = np.nonzero(predictions == y)[0]

i = 0
for c in correct[:6]:
    plt.subplot(3, 2, i+1)
    plt.imshow(x_test[c].reshape(28, 28), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y[c]))
    plt.tight_layout()
    i += 1
