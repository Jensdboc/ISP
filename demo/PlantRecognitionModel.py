import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

VERSION = "_v0_train_20"
print("Model" + VERSION)

# Define paths
path = "/user/gent/450/vsc45058/ISP/Data/Dataset1/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
path_train = path + "/train"
path_valid = path + "/valid"

# Construct model
classes = [x for x in os.listdir(path_train) if os.path.isdir(os.path.join(path_train, x))]
num_classes = len(classes)
print(f"Number of classes is {num_classes}\n")

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes))

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define the data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

batch_size = 32  # Adjust as needed
train_generator = train_datagen.flow_from_directory(
    path_train,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='sparse'  # Assuming a classification task
)
validation_generator = validation_datagen.flow_from_directory(
    path_valid,
    target_size=(256, 256),
    batch_size=batch_size,
    class_mode='sparse'  # Assuming a classification task
)

# Training loop
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Print history (optional)
print(history.history)

# Plot validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("/user/gent/450/vsc45058/ISP/acc_curves" + VERSION + ".png")

plt.plot(epochs, loss, 'green', label='Training loss')
plt.plot(epochs, val_loss, 'black', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("/user/gent/450/vsc45058/ISP/loss_curves" + VERSION + ".png")

model.save("/user/gent/450/vsc45058/ISP/model" + VERSION + ".h5")
model.save("/user/gent/450/vsc45058/ISP/model" + VERSION + ".keras")
