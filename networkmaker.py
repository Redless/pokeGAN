from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Reshape, UpSampling2D, Flatten

noiseDim = 100

inputLayer = Input(shape=(noiseDim,))

generator = Dense(256,activation="relu")(inputLayer)
generator = Dense(128,activation="relu")(generator)
generator = Dense(64,activation="relu")(generator)
generator = Reshape((8,8,1))(generator)
generator = UpSampling2D(4)(generator)
generator = Conv2D(100,7,padding="same",activation="relu")(generator)
generator = UpSampling2D(3)(generator)
generator = Conv2D(50,7,padding="same",activation="relu")(generator)
generator = Conv2D(30,7,padding="same",activation="relu")(generator)
generator = Conv2D(4,7,padding="same",activation="sigmoid")(generator)
generator = Model(inputLayer,generator)
generator.compile(loss='binary_crossentropy', optimizer="adam")

print("generator constructed...")
generator.save("generator")
print("generator saved.")

inputLayer = Input(shape=(96,96,4))

discriminator = Flatten()(inputLayer)
discriminator = Dense(128,activation="relu")(discriminator)
discriminator = Dense(64, activation="relu")(discriminator)
discriminator = Dense(1, activation='sigmoid')(discriminator)

discriminator = Model(inputLayer,discriminator)
discriminator.compile(loss='binary_crossentropy', optimizer="adam")

print("discriminator constructed...")
discriminator.save("discriminator")
print("discriminator saved")
