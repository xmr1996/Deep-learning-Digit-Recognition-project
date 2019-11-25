import numpy as np
from PIL import Image
import os
import glob
import pickle

from tensorflow_core.python.keras.layers.core import Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from tensorflow_core.python.keras import models
from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.pooling import MaxPooling2D
from tensorflow_core.python.keras.callbacks import TensorBoard


def build_network():
    network = models.Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu'))
    network.add(MaxPooling2D((2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu'))
    network.add(Flatten())
    network.add(Dense(64, activation='relu'))
    # network.add(Dense(32, activation='relu'))
    network.add(Dense(10, activation='softmax'))

    network.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    return network


network = build_network()

def prepare_data(path):

    sample = []
    label = []

    for png in glob.glob(path + '\*.png'):
        label.append(int(os.path.basename(png).split('_')[0]))

        img = Image.open(png)
        arr = np.array(img) / 255
        sample.append(arr[np.newaxis, :, :])

    input = np.concatenate(sample, axis=0)
    label = np.array(label)
    input = input[:, :, :, np.newaxis]

    return (input, label)


#training the model
(input,label) = prepare_data(r"Procced_img")

index = np.random.permutation(np.arange(958))
input = input[index]
label = label[index]

print(input.shape)
print(label.shape)

# tf_board = TensorBoard(log_dir='logs')

his = network.fit(input[:920], label[:920],
            validation_data=[input[920:], label[920:]],
            epochs=50, batch_size=128)

#Construct the grapths
history_dict = his.history
# loss value for training data
loss_values = history_dict['loss']
# loss value for validation data
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)

# blue dots for training loss
plt.plot(epochs, loss_values, 'bo', label='Training Loss')
# solid blue line for validation loss
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss values')
plt.xlabel('Epochs')
plt.ylabel('Epochs')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()


#Here is the code for testing
(test_images,test_labels) = prepare_data(r"test_img")

print(test_images.shape)
print(test_labels.shape)

print('test begins here')
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("Model accuracy: ", test_acc)
print("Model loss", test_loss)


# input = input[:,:,:,np.newaxis]
# with open('input_data.pickle', 'wb') as file:
#     pickle.dump(input, file)

# with open('label.pickle', 'wb') as file:
#     pickle.dump(label, file)

# with open('input_data.pickle', 'rb') as file:
#     input = pickle.load(file)
#

# datagen = image.ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
#
# datagen.fit(input)
# network.fit_generator(datagen.flow(input[:150], label[:150], batch_size=32),
#                       validation_data=datagen.flow(input[150:], label[150:]),
#                       validation_steps=50,
#                       steps_per_epoch=200,
#                       epochs=100)

# with open('label.pickle', 'rb') as file:
#     label = pickle.load(file)