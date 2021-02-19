"""
Very basic feed fw network with two heads
"""

from tensorflow import keras

def build_simple_model(input_size, output_size):

    dIn = keras.layers.Input(shape=input_size)
    lr = keras.layers.Dense(10, activation='relu')(dIn)
    lr = keras.layers.Dense(10, activation='relu')(lr)
    # Policy
    pol = keras.layers.Dense(10, activation='relu')(lr)
    pol = keras.layers.Dense(output_size)(pol)

    # ValueF
    vf = keras.layers.Dense(10)(lr)
    vf = keras.layers.Dense(1)(vf)

    return keras.Model(dIn, [pol, vf])