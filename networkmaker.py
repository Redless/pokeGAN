from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape, UpSampling2D, Flatten, Dropout

noiseDim = 640

inputLayer = Input(shape=(noiseDim,))

generator = Dense(128,activation="relu")(inputLayer)
generator = Reshape((4,4,40))(inputLayer)
generator = Conv2D(50,2,padding="same",activation="relu")(generator)
generator = UpSampling2D(4)(generator)
generator = Conv2D(40,3,padding="same",activation="relu")(generator)
generator = UpSampling2D(3)(generator)
generator = Conv2D(30,5,padding="same",activation="relu")(generator)
generator = UpSampling2D(2)(generator)
generator = Conv2D(20,7,padding="same",activation="relu")(generator)
generator = Conv2D(4,4,padding="same",activation="sigmoid")(generator)
generator = Model(inputLayer,generator)
generator.compile(loss='binary_crossentropy', optimizer="adam")

print(generator.summary())
print("generator constructed...")
generator.save("generator")
print("generator saved.")

inputLayer = Input(shape=(96,96,4))

discriminator = Conv2D(8,5,padding="same",activation="relu")(inputLayer)
discriminator = Flatten()(discriminator)
discriminator = Dense(32,activation="relu")(discriminator)
discriminator = Dense(1, activation='sigmoid')(discriminator)

discriminator = Model(inputLayer,discriminator)
discriminator.compile(loss='binary_crossentropy', optimizer="adam")

print(discriminator.summary())
print("discriminator constructed...")
discriminator.save("discriminator")
print("discriminator saved")

