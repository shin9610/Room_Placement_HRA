from keras.layers import Lambda, Input, Convolution2D
from keras.layers.core import Dense, Flatten
from keras.models import Model

floatX = 'float32'


def build_dense(state_shape, nb_units, nb_actions, nb_channels, remove_features=False):
    # featuresの処理がよく分からん
    # とにかくここで生成されるモデルがreward分ある。
    if remove_features:
        state_shape = state_shape[: -1] + [state_shape[-1] - nb_channels + 1]

    # 入力層：(4, 28, 28)
    input_dim = tuple(state_shape)
    states = Input(shape=input_dim, dtype=floatX, name='states')
    flatten = Flatten()(states)
    
    # 中間層
    hid = Dense(output_dim=nb_units, init='he_uniform', activation='relu', name='hidden')(flatten)
    
    # 出力層
    out = Dense(output_dim=nb_actions, init='he_uniform', activation='linear', name='out')(hid)
    return Model(input=states, output=out)

def build_cnn(state_shape, nb_units, nb_actions, nb_channels, remove_features=False):
    # CNN版ではない　→　修正必要
    # if remove_features:
    #     state_shape = state_shape[: -1] + [state_shape[-1] - nb_channels + 1]

    # 入力層
    input_dim = tuple(state_shape)
    state_input = Input(shape=input_dim, name='state')
    
    # 中間層
    x = Convolution2D(16, (4, 4), activation='relu', strides=(2, 2), padding='same', data_format='channels_first')(
        state_input)
    x = Convolution2D(32, (2, 2), activation='relu', strides=(1, 1), padding='same', data_format='channels_first')(x)
    x = Convolution2D(32, (2, 2), activation='relu', strides=(1, 1), padding='same', data_format='channels_first')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    # 出力層
    out = Dense(nb_actions, activation='linear', name='main_output')(x)
    
    return Model(input=state_input, output=out)

    