from os import listdir
import random
import numpy as np
from imageio import imread, imwrite
from keras.models import load_model
from keras.layers import Input
from keras import Model

spritelist = listdir("every_pokemon_sprite/sprites")
noiseDim = 640

discriminator = load_model("discriminator")
discriminator.trainable=False
generator = load_model("generator")
inlayer = Input((noiseDim,))
full_gan = discriminator(generator(inlayer))
full_gan = Model(inlayer,full_gan)
full_gan.compile(loss='binary_crossentropy', optimizer="adam")

def train_discriminator(batchsize):
    discriminator.trainable = True
    noise = np.random.normal(0, 1, size=[batchsize, noiseDim])
    
    inarr = np.empty((batchsize*2,96,96,4))
    for i in range(batchsize):
        sprite2load = random.choice(spritelist)
        inarr[i] = imread("every_pokemon_sprite/sprites/"+sprite2load,format="png")/255.

    inarr[batchsize:,:,:,:] = generator.predict(noise)

    key = np.zeros(batchsize*2)
    key[:batchsize] = .9

    ourloss = discriminator.train_on_batch(inarr,key)
    return "discriminator loss: "+str(ourloss)

def train_generator(batchsize):
    discriminator.trainable = False
    noise = np.random.normal(0, 1, size=[batchsize, noiseDim])
    ourloss = full_gan.train_on_batch(noise,np.ones(batchsize))
    return "generator loss: "+str(ourloss)

def test_generator(epoch,numimgs):
    noise = np.random.normal(0, 1, size=[numimgs,noiseDim])
    results = (generator.predict(noise)*255).astype("uint8")
    for i in range(numimgs):
        imwrite('generated_images/'+str(epoch)+"epoch"+str(i)+".png",results[i,:])

def save_gnd(epoch):
    discriminator.save("trainedmodels/discrim"+str(epoch))
    generator.save("trainedmodels/gener"+str(epoch))
    discriminator.save("discriminator")
    generator.save("generator")

for epoch in range(50):
    print("-"*10+"[ E P O C H "+str(epoch)+"]"+"-"*10)
    for j in range(100):
        print(train_discriminator(1))
        for i in range(10):
            print(train_generator(1))
    save_gnd(epoch)
    test_generator(epoch,5)
