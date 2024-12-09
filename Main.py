import tensorflow as tf
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Creating the generator model
def make_generator_model():
    generator = tf.keras.Sequential()
    generator.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    generator.add(tf.keras.layers.Reshape([7, 7, 256]))
    generator.add(tf.keras.layers.BatchNormalization())
    generator.add(tf.keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", activation="relu"))
    generator.add(tf.keras.layers.BatchNormalization())
    generator.add(tf.keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh"))
    return generator

# Creating the discriminator model
def make_discriminator_model():
    discriminator = tf.keras.Sequential()
    discriminator.add(tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2), input_shape=[28, 28, 1]))
    discriminator.add(tf.keras.layers.Dropout(0.4))
    discriminator.add(tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same", activation=tf.keras.layers.LeakyReLU(0.2)))
    discriminator.add(tf.keras.layers.Dropout(0.4))
    discriminator.add(tf.keras.layers.Flatten())
    discriminator.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return discriminator

# Instantiate the models
gen = make_generator_model()
disc = make_discriminator_model()

# Directory for saving checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Loading and preparing the dataset
(train_images, train_labels), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32')
train_images = (train_images - 127.5) / 127.5  # This normalizes the images as [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batching and shuffling the data
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Create the checkpoint object after the models are defined
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=gen,
                                 discriminator=disc)

# Define training loop
EPOCHS = 60
noise_dim = 100
images_to_gen = 20

# Setting the seed (random)
seed = tf.random.normal([images_to_gen, noise_dim])

@tf.function
def training_steps(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = gen(noise, training=True)

        real_output = disc(images, training=True)
        fake_output = disc(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            training_steps(image_batch)

        # Produces images
        clear_output(wait=True)
        generate_and_save_images(gen, epoch + 1, seed)

        # Saves every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 10, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# Training the model
train(train_dataset, EPOCHS)