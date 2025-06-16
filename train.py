import tensorflow as tf
import tensorflow_datasets as tfds
print(tf.version.VERSION)

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs are available: {gpus}")
else:
    print("No GPUs found.")
    import tensorflow as tf
import tensorflow_datasets as tfds

# Load the dataset and split into train (70%), val (20%), and test (10%)
splits = ['train[:70%]', 'train[70%:90%]', 'train[90%:]']
(train_ds, val_ds, test_ds), info = tfds.load('plant_village', with_info=True, as_supervised=True, split=splits)

# Define image size and batch size
IMG_SIZE = 180
BATCH_SIZE = 64

# Data augmentation function
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    return image, label

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Apply augmentation only to the training dataset
train_ds = train_ds.map(augment).map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#train_ds = train_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Apply preprocessing to validation and test datasets
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Print dataset information
print(info)
import matplotlib.pyplot as plt

# Function to display a batch of images and their labels
def visualize_sample_data(dataset, class_names, sample_size=9):
    plt.figure(figsize=(10, 10))
    
    for images, labels in dataset.take(1):
        for i in range(sample_size):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()

# Retrieve class names
class_names = info.features['label'].names

# Visualize a sample of 9 images from the train dataset
visualize_sample_data(train_ds, class_names)

# Print class names with their corresponding indices
for index, class_name in enumerate(class_names):
    print(f'Class index: {index}, Class name: {class_name}')
    # Load the EfficientNetB3 base model
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                                  include_top=False,
                                                  weights='imagenet')

base_model.trainable = False  # Freeze the base model

# Define the model using Functional API
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)  # Ensure base_model runs in inference mode
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(38, activation='softmax')(x)

# Build the model
model = tf.keras.Model(inputs, outputs)

# print summary
model.summary()
# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define EarlyStopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=50,  # Number of epochs to wait for improvement
    restore_best_weights=True,
    verbose=1
)

# Define ModelCheckpoint callback for both best model and latest model
model_checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

model_checkpoint_latest = tf.keras.callbacks.ModelCheckpoint(
    'latest_model.keras',
    save_weights_only=True,
    mode='auto',
    verbose=1
)

# Define LearningRateScheduler callback with exponential decay
def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    scheduler,
    verbose=1
)

# # Define TensorBoard callback for monitoring
# tensorboard = tf.keras.callbacks.TensorBoard(
#     log_dir='./logs',
#     histogram_freq=1,
#     write_graph=True,
#     write_images=False,
#     update_freq='epoch'
# )

# Train the model with the defined callbacks
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=250,
    callbacks=[early_stopping, model_checkpoint_best, model_checkpoint_latest, lr_scheduler]
)

# Save the entire model as a `.keras` zip archive
model.save('final_model.keras')