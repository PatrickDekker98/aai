import pickle , gzip , os
from urllib import request
from pylab import imshow , show , cm
import numpy as np
import tensorflow as tf

url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
if not os.path.isfile("mnist.pkl.gz"):
    request.urlretrieve(url , "mnist.pkl.gz")

f = gzip.open('mnist.pkl.gz', 'rb')
train_set , valid_set , test_set = pickle.load(f, encoding ='latin1')
f.close()

def get_image ( number ):
    (X, y) = [img[ number ] for img in train_set ]
    return (np.array(X), y)

def view_image ( number ):
    (X, y) = get_image( number )
    print(" Label : %s" % y)
    imshow(X.reshape (28 ,28) , cmap=cm.gray)
    show()

nn = tf.keras.models.Sequential()
nn.add(tf.keras.layers.Flatten())
nn.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
nn.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
nn.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
nn.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

train_y = list()
for y in train_set[1]:
    ya = [0 for i in range(10)] 
    ya[y] = 1
    train_y.append(np.array(ya))

print(len(train_set[0]))
print(len(train_y))

nn.fit(train_set[0], train_set[1], epochs=3)

val_loss, val_acc = nn.evaluate(valid_set[0], valid_set[1])
print(val_loss, val_acc)
val_loss, val_acc = nn.evaluate(test_set[0], test_set[1])
print(val_loss, val_acc)
