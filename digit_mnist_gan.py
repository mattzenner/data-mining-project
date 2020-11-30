# https://towardsdatascience.com/writing-your-first-generative-adversarial-network-with-keras-2d16fd8d4889

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import time as t
from time import time

BATCH = 256

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.generator = self.build_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        validity = self.discriminator(img)
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def save_model(self, g_loc, d_loc):
        self.generator.save(g_loc)
        self.discriminator.save(d_loc)

    def load_model(self, g_loc, d_loc):
        self.generator = keras.models.load_model(g_loc)
        self.discriminator = keras.models.load_model(d_loc)

    def build_generator(self):
            model = Sequential()
            model.add(Dense(256, input_dim=self.latent_dim))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(1024))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Dense(np.prod(self.img_shape), activation='tanh'))
            model.add(Reshape(self.img_shape))
            model.summary()
            noise = Input(shape=(self.latent_dim,))
            img = model(noise)
            return Model(noise, img)


    def build_discriminator(self):
            model = Sequential()
            model.add(Flatten(input_shape=self.img_shape))
            model.add(Dense(512))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(256))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dense(1, activation='sigmoid'))
            model.summary()
            img = Input(shape=self.img_shape)
            validity = model(img)
            return Model(img, validity)

#https://github.com/keras-team/keras/issues/2708
    def train(self, epochs, batch_size=128, sample_interval=50, print_interval=50):
            X_mnist = mnist.load_data()
            X_train, _ = train_test_split(pd.read_csv('D://gan//asl_mnist_style.csv')) # mnist.load_data()
            #mnist.load_data() returns a 2-tuple of pixel data, labels so var _ becomes labels arr
            labels = X_train['label']
            del X_train['label']
            X_train = X_train.to_numpy()
            X_train = X_train / 127.5 - 1.
            #X_train = np.expand_dims(X_train, axis=3)
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            print('X_train type: ', type(X_train))
            print('X_train shape: ', X_train.shape)
            print('X_train size: ', X_train.size)
            # print('MNIST: ')
            # print('type: ', type(X_mnist))
            # print('length: ', len(X_mnist))
            # print('first: ', X_mnist[0])
            # print('second: ', X_mnist[1])
            # print(X_mnist[0][0])
            last = time()
            for epoch in range(epochs):
                idx = np.random.randint(0, len(X_train), batch_size) #X_train.shape[0]
                #imgs = X_train.iloc[idx, :]
                imgs = X_train[idx] # [np.reshape(img, (28, 28, 1)) for img in X_train[idx]]
                # print('imgs: ')
                # print('type: ', type(imgs))
                # print('length: ', len(imgs))
                # print(imgs)
                imgs = np.reshape(imgs, (batch_size, 28, 28, 1))
                # for img in imgs:
                #     #print(img)
                #     #print(type(img))
                #     #print(len(img))
                #     img = np.reshape(img, (28, 28, 1))
                #     #print(type(img))
                #     #print(img.shape)
                # imgs = [np.reshape(img[0], (28, 28, 1)) for img in imgs]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                d_loss_real = self.discriminator.train_on_batch(imgs, valid)
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                g_loss = self.combined.train_on_batch(noise, valid)
                if epoch % print_interval == 0:
                    print ("{} [D loss: {:.4}, acc: {:.4}%] [G loss: {:.4}] {:.4} minutes.".format(epoch, d_loss[0], str(100.0*d_loss[1]), g_loss, (int(time() - last))/60))
                    last = time()
                if epoch % sample_interval == 0:
                    self.sample_images(epoch)


    def sample_images(self, epoch):
            r, c = 5, 5
            noise = np.random.normal(0, 1, (r * c, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig("D://gan//images_asl_1//{}.png".format(epoch))
            plt.close()


    def generate_images(self):
            r, c = 5, 5
            noise = np.random.normal(0, 1, (r * c, self.latent_dim))
            gen_imgs = self.generator.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1
            # fig.savefig("D://gan/{}.png".format(epoch))
            plt.show()
            plt.close()

# if __name__ == '__main__':
gan = GAN()

epochs = 30000
gan.train(epochs=epochs, batch_size=256, sample_interval=int(epochs/10), print_interval=int(epochs/100))
gan.save_model('D://gan//saved_model_asl_1//generator', 'D://gan//saved_model_asl_1//discriminator')
print('Generator summary:')
gan.generator.summary()
print('Discriminator summary: ')
gan.discriminator.summary()
print('Combined summary: ')
gan.combined.summary()

# gan.load_model('D://gan//saved_model_1//generator', 'D://gan//saved_model_1//discriminator')

print("Ready to generate!")
for x in range(3):
    print(x + 1)
    gan.generate_images()
