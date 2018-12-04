"""
    Les réseaux de neurones plus profonds sont capables de créer des structures plus complexes dans les données d’entrée.

     MNIST est un ensemble d’images 28 x 28 pixels composées à la main, 
     avec 60 000 images d’entraînement et 10 000 images test. 
     Les jeux de données CIFAR-10 et CIFAR-100 consistent en des images de 32 x 32 pixels en 10 et 100 classes, respectivement. 
     Les deux ensembles de données contiennent 50 000 images de formation et 10 000 images de test. 
     
     Le repo de github pour Keras contient un exemple de réseaux de neurones convolutionnels (CNN) pour MNIST et CIFAR-10 .

    Mon objectif est de créer un CNN utilisant Keras pour CIFAR-100 et adapté à une instance Amazon Web Services (AWS)  EC2


"""



import keras

import numpy as np
import os
import matplotlib.pyplot as plt
import chainer
basedir = './src/cnn/images'

CIFAR10_LABELS_LIST = [
    'airplane', 
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]
 
train, test = chainer.datasets.get_cifar10()


# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

print('len(train), type ', len(train), type(train))
print('len(test), type ', len(test), type(test))

print('train[0]', type(train[0]), len(train[0]))
 
x0, y0 = train[0]
print('train[0][0]', x0.shape, x0)
print('train[0][1]', y0.shape, y0, '->', CIFAR10_LABELS_LIST[y0])


def plot_cifar(filepath, data, row, col, scale=3., label_list=None):
    fig_width = data[0][0].shape[1] / 80 * row * scale
    fig_height = data[0][0].shape[2] / 80 * col * scale
    fig, axes = plt.subplots(row, 
                             col, 
                             figsize=(fig_height, fig_width))
    for i in range(row * col):
        # train[i][0] is i-th image data with size 32x32
        image, label_index = data[i]
        image = image.transpose(1, 2, 0)
        r, c = divmod(i, col)
        axes[r][c].imshow(image)  # cmap='gray' is for black and white picture.
        if label_list is None:
            axes[r][c].set_title('label {}'.format(label_index))
        else:
            axes[r][c].set_title('{}: {}'.format(label_index, label_list[label_index]))
        axes[r][c].axis('off')  # do not show axis value
    plt.tight_layout()   # automatic padding between subplots
    plt.savefig(filepath)
    plt.show()


plot_cifar(os.path.join(basedir, 'cifar10_plot.png'), train, 4, 5, 
           scale=4., label_list=CIFAR10_LABELS_LIST)

plot_cifar(os.path.join(basedir, 'cifar10_plot_more.png'), train, 10, 10, 
           scale=4., label_list=CIFAR10_LABELS_LIST)
           
# print('train[0]', type(train[0]), len(train[0]))
 
# x0, y0 = train[0]
# print('train[0][0]', x0.shape, x0)
# print('train[0][1]', y0.shape, y0, '->', CIFAR10_LABELS_LIST[y0])

# Create model

"""

    La première pile comporte deux couches convolutives de 128 neurones. 
    Activation: formule sigmoid

    La mise en commun se fait en regardant une grille (ici une grille 2x2) et en n'utilisant que le maximum. 
    
    Cela supprime le nombre de paramètres utilisés et ne laisse que les plus importants pour éviter les surajustements, 
    avec l'avantage supplémentaire de réduire la quantité de mémoire nécessaire au réseau. 
    Une suppression est ensuite utilisée pour éviter davantage de surajustement en fixant à zéro 
    
    Les débits d'entrée fractionnaires égaux ou inférieurs à la valeur de la perte. 
    Cela supprimera les neurones du réseau, ne laissant que les connexions neuronales avec des poids plus lourds.
"""
# model = Sequential()
# model.add(Conv2D(128, (3, 3), padding='same',
#     input_shape=x_train.shape[1:]))
# model.add(Activation('elu')) # ReLU
# model.add(Conv2D(128, (3, 3)))
# model.add(Activation('elu')) # ReLU
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.1))

"""
    Note: L'unité linéaire exponentielle ( ELU; Clevert et al. 2015 ) est une autre fonction d'activation qui accélère le processus d'apprentissage et crée un réseau de neurones avec une précision élevée sur les ReLU, en particulier sur le jeu de données ICRA 100 
"""

"""
    Passons aux deux piles suivantes avec regroupement 2x2 et abandons après chacune
"""

# model.add(Conv2D(256, (3, 3), padding='same'))
# model.add(Activation('elu'))
# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# model.add(Conv2D(512, (3, 3), padding='same'))
# model.add(Activation('elu'))
# model.add(Conv2D(512, (3, 3)))
# model.add(Activation('elu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

"""
    Le réseau de neurones doit en fin de compte sortir la probabilité des différentes classes d’un tableau. 
    Après les piles de convolution, les probabilités doivent être aplaties en un vecteur de caractéristiques 1D. 
    Les couches denses sont des couches entièrement connectées qui appliquent des transformations et modifient les dimensions.
    La couche dense finale doit avoir la même longueur que le nombre de classes et donner la probabilité de chaque classe.
"""

# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('elu'))
# model.add(Dropout(0.5))
# model.add(Dense(100))
# model.add(Activation('softmax')) # Multi Proba by classes