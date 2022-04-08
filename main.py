from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''Load the datasets'''
dataset_path = 'https://raw.githubusercontent.com/ajailani4/leaf-classification/main/leaf.csv?token=GHSAT0AAAAAABPBYOL4MOQTZFRIRJBINWA4YSW2LDQ'
dataset = read_csv(dataset_path, header=None)

species_names = ['Quercus suber', 'Salix atrocinera', 'Populus nigra', 'Alnus sp.', 'Quercus robur', 'Crataegus monogyna', 'Ilex aquifolium', 
'Nerium oleander', 'Betula pubescens', 'Tilia tomentosa', 'Acer palmatum', 'Celtis sp.', 'Corylus avellana', 'Castanea sativa', 
'Populus alba', 'Primula vulgaris', 'Erodium sp.', 'Bougainvillea sp.', 'Arisarum vulgare', 'Euonymus japonicus', 'Ilex perado ssp. azorica', 
'Magnolia soulangeana', 'Buxus sempervirens', 'Urtica dioica', 'Podocarpus sp.', 'Acca sellowiana', 'Hydrangea sp.', 'Pseudosasa japonica', 
'Magnolia grandiflora', 'Geranium sp.']

'''Preprocess the data'''
# Split into input (images) and output (species) columns
images, species = dataset.values[:, 2:], dataset.values[:, 0]
images = images.astype('float32')

# Normalize labels
species = LabelEncoder().fit_transform(species)

# Split into train and test datasets
train_images, test_images, train_species, test_species = train_test_split(images, species, test_size=0.2)
print('Train images\n', train_images)
print('\nTrain species\n\n', train_species)

'''Building the model'''
n_features = train_images.shape[1]

# Setup the layers
model = tf.keras.Sequential([
  tf.keras.layers.Dense(120, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)),
  tf.keras.layers.Dense(110, activation='relu', kernel_initializer='he_normal'),
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

'''Make predictions for a single data'''
print('\nTest images\n', test_images)
print('\nTest species\n', test_species)

i = 0
img = test_images[i]
img = (np.expand_dims(img, 0))
predictions = model.predict(img)
print('\nPredictions:', predictions)
print('\nTested image:', test_images[i])
print(
    '\nPredicted species: {} ({:2.0f}%) | Actual species: {}'.format(
        species_names[np.argmax(predictions[i])],
        100*np.max(predictions[i]),
        species_names[test_species[i]]
    )
)
