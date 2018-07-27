from keras.models import Model
from keras.layers import Input, Dense, Con, Reshape

noiseDim = 100

inputLayer = Input(shape=(noiseDim,))

generator = Dense(64,activation="ReLu")(inputLayer)
generator = Reshape((8,8))(generator)
generator = UpSampling2D(4)(generator)
generator = Conv2D(10,5,activation="ReLu")(generator)
generator = UpSampling2D(3)(generator)
generator = Conv2D(4,5,activation="Sigmoid")(generator)
