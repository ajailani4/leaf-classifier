from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''Load the datasets'''
dataset_path = 'https://raw.githubusercontent.com/ajailani4/leaf-classification/main/leaf.csv?token=GHSAT0AAAAAABPBYOL4MOQTZFRIRJBINWA4YSW2LDQ'
dataset = read_csv(dataset_path, header=None)

'''Preprocess the data'''
# Split into input (images) and output (species) columns
images, species = dataset.values[:, 1:], dataset.values[:, 0]
images = images.astype('float32')

# Normalize labels
species = LabelEncoder().fit_transform(species)

# Split into train and test datasets
train_images, test_images, train_species, test_species = train_test_split(images, species, test_size=0.1)
print('Train images\n', train_images)
print('\nTrain species\n', train_species)

'''Building the model'''
n_features = train_images.shape[1]

# Setup the layers
model = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)),
  tf.keras.layers.Dense(80, activation='relu', kernel_initializer='he_normal'),
  tf.keras.layers.Dense(30, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

'''Train the model'''
# Feed the model
model.fit(train_images, train_species, epochs=150, verbose=2)

# Evaluate accuracy
train_loss, train_acc = model.evaluate(train_images, train_species)
test_loss, test_acc = model.evaluate(test_images, test_species)
print('\nTrain accuracy: %.2f%%' % (train_acc*100))
print('\nTest accuracy: %.2f%%' % (test_acc*100))

# Make a prediction
print('\nTest images\n', test_images)
print('\nTest species\n', test_species)

img = test_images[0]
img = (np.expand_dims(img, 0))
predictions = model.predict(img)
print(predictions)
print(
    '\nPredicted species: {} ({:2.0f}%) | Actual species: {}'.format(
        np.argmax(predictions[0]),
        100*np.max(predictions[0]),
        'test'
    )
)