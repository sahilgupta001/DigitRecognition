from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.python.keras.optimizers import RMSprop

(mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()


train_images = mnist_train_images.reshape(60000, 784)
test_images = mnist_test_images.reshape(10000,784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255.0
test_images /= 255.0

train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels = keras.utils.to_categorical(mnist_test_labels, 10)



# Visualising the data that we have...
# print("\nNow we are visualising the data...")
# def displaySample(num):
#     print(train_labels[num])
#     label = train_labels[num].argmax(axis = 0)
#     image = train_images[num].reshape([28, 28])
#     plt.title('Sample: %d Label: %d'% (num, label))
#     plt.imshow(image, cmap = plt.get_cmap('gray_r'))
#     plt.show()
#
# displaySample(1234)


model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))


model.compile(loss = "categorical_crossentropy", optimizer = RMSprop(), metrics = ['accuracy'])
history = model.fit(train_images, train_labels, batch_size = 100, epochs = 20, verbose = 2, validation_data = (test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose = 0)
print("Test Loss: ", score[0])
print("Test accuracy: ", score[1])

for x in range(1000):
    test_image = test_images[x,:].reshape(1, 784)
    predicted_cat = model.predict(test_image).argmax()
    label = test_labels[x].argmax()
    if (predicted_cat != label):
        plt.title('Prediction: %d Label: %d'%(predicted_cat, label))
        plt.imshow(test_image.reshape([28, 28]), cmap = plt.get_cmap('gray_r'))
        plt.show()


