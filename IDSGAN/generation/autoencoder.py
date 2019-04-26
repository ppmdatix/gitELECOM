from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers




def create_auto_encoder(layers_size=[100,50,25,]):
    adam = Adam()

    input_img= Input(shape=(layers_size[0],))
    layers_size[]
    # encoded and decoded layer for the autoencoder
    encoded = Dense(units=layers_size[], activation='relu')(input_img)
    encoded = Dense(units=layers_size[], activation='relu')(encoded)
    encoded = Dense(units=layers_size[], activation='relu')(encoded)
    decoded_1 = Dense(units=layers_size[], activation='relu')(encoded)
    decoded = Dense(units=layers_size[], activation='relu')(decoded_1)
    decoded = Dense(units=layers_size[], activation='sigmoid')(decoded)

    # Building autoencoder
    autoencoder=Model(input_img, decoded)

    #extracting encoder
    encoder = Model(input_img, encoded)

    # extracting decoder
    decoder = Model(decoded_1, decoded)

    # compiling the autoencoder
    autoencoder.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the noise trained data to the autoencoder
    autoencoder.fit(X_train_noisy, X_train_noisy,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test_noisy, X_test_noisy))

    # reconstructing the image from autoencoder and encoder
    encoded_imgs = encoder.predict(X_test_noisy)
    predicted = autoencoder.predict(X_test_noisy)

    # plotting the noised image, encoded image and the reconstructed image
    plt.figure(figsize=(40, 4))
    for i in range(10):
        # display original images

        ax = plt.subplot(4, 20, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display noised images
        ax = plt.subplot(4, 20, i + 1+20)
        plt.imshow(X_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display encoded images
        ax = plt.subplot(4, 20, 2*20+i + 1 )
        plt.imshow(encoded_imgs[i].reshape(8,4))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction images
        ax = plt.subplot(4, 20, 3*20 +i+ 1)
        plt.imshow(predicted[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)


    plt.show()