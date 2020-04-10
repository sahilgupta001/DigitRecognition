import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sess = tf.compat.v1.InteractiveSession()
mnist = tf.keras.datasets.mnist

(x_train, y_train) , (x_test, y_test) = mnist.load_data()

train_images = x_train.reshape(60000, 784)
test_images = x_test.reshape(10000, 784)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

x_train, x_test = train_images / 255.0, test_images / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def plot(num):
    maxValue = y_train[0].argmax(axis = 0)
    image = x_train[num].reshape([28, 28])
    plt.title("Sample image from the train set...")
    plt.imshow(image, cmap = plt.get_cmap('gray_r'))
    plt.show()
# plot(1234)


input_images = tf.compat.v1.placeholder(tf.float32, shape = [None, 784])
target_labels = tf.compat.v1.placeholder(tf.float32, shape = [None, 10])

hidden_nodes = 512
input_weights = tf.Variable(tf.random.truncated_normal([784, hidden_nodes]))
input_biases = tf.Variable(tf.zeros([hidden_nodes]))

hidden_weights = tf.Variable(tf.random.truncated_normal([hidden_nodes, 10]))
hidden_biases = tf.Variable(tf.zeros([10]))

input_layer = tf.matmul(input_images, input_weights)
hidden_layer = tf.nn.relu(input_layer + input_biases)
digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases


loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = digit_weights, labels = target_labels))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(loss_function)


cross_prediction = tf.equal(tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))

accuracy = tf.reduce_mean(tf.cast(cross_prediction, tf.float32))


tf.global_variables_initializer().run()

EPOCH = 20
BATCH_SIZE = 100
TRAIN_DATASIZE,_ = x_train.shape
PERIOD = TRAIN_DATASIZE//BATCH_SIZE

# for e in range(EPOCH):
#     idxs = np.random.permutation(TRAIN_DATASIZE)
#     X_RANDOM = x_train[idxs]
#     Y_RANDOM = y_train[idxs]
#     for i in range(PERIOD):
#         batch_X = X_RANDOM[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
#         batch_Y = Y_RANDOM[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
#         optimizer.run(feed_dict= {input_images: batch_X, target_labels: batch_Y})
#     print("Training EPOCH: ", str(e + 1))
#     print("Accuracy: " + str(accuracy.eval(feed_dict = {input_images: x_test, target_labels: y_test})))


for x in range(100):
    x_train_one = x_test[x,:].reshape(1, 784)
    y_train_one = y_test[x,:]
    # Convert the red hot label to an image
    label = y_train_one.argmax()
    pred = sess.run(digit_weights, feed_dict = {input_images: x_train_one}).argmax()
    if(pred != label):
        plt.title('Prediction: %d Label: %d'%(pred, label))
        plt.imshow(x_train_one.reshape([28,28]), cmap = plt.get_cmap('gray_r'))
        plt.show()
