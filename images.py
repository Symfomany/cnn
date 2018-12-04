"""
The CIFAR-10 and CIFAR-100 are labeled subsets of the 80 million tiny images dataset. 
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 

https://www.cs.toronto.edu/~kriz/cifar.html

Ressource:
https://andrewkruger.github.io/projects/2017-08-05-keras-convolutional-neural-network-for-cifar-100

"""

from keras.datasets import cifar10

# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

# """
# x_train, x_test: uint8 array of RGB image data with shape (num_samples, 3, 32, 32) or (num_samples, 32, 32, 3) 
# based on the image_data_format backend setting of either channels_first or channels_last respectively.
# """

# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# print('y_train shape:', y_train.shape)
# print(y_train.shape[0], 'train samples')
# print(y_test.shape[0], 'test samples')


def cifar10_extract(label = 'cat'):
    # acceptable label
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

    target_label = labels.index(label)

    (x_train, t_train), (x_test, t_test) = cifar10.load_data(label_mode='fine')

    t_target = t_train==target_label
    t_target = t_target.reshape(t_target.size)

    x_target = x_train[t_target]
    
    print('extract {} labeled images, shape(5000, 32, 32, 3)'.format(label))
    return x_target


cifar10_extract();