from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, concatenate, Reshape, Dropout, LSTM, TimeDistributed, Concatenate, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow import keras

def getAutoEncoder(input_shape):

    # The encoding process
    input_img = Input(shape=input_shape)  

    ############
    # Encoding #
    ############

    # Conv1 #
    x = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    # Conv2 #
    x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x) 

    # Conv 3 #
    x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

    
    
    ###############
    # Forecasting #
    ###############
    x=Dense(units = 45, activation = 'relu')(encoded)
    forecast = Dense(units = 45, activation = 'relu')(x)
    
    

    ############
    # Decoding #
    ############

    # DeConv1
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(forecast)
    x = UpSampling2D((2, 2))(x)

    # DeConv2
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Deconv3
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    return Model(input_img, decoded)
    
    
def getSequentialAutoEncoder(input_shape):

    inputs = []
    encoded = []
    for i in range(input_shape[2]):
        # The encoding process
        inputs.append(Input(shape=(input_shape[0], input_shape[1], 1)))

        ############
        # Encoding #
        ############

        # Conv1 #
        x = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same')(inputs[i])
        x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)

        # Conv2 #
        x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size = (2, 2), padding='same')(x) 

        # Conv 3 #
        x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same')(x)
        encoded.append(MaxPooling2D(pool_size = (2, 2), padding='same')(x))

        
    
    ###############
    # Forecasting #
    ###############
    x = Flatten()(encoded[0])
    x=Dense(units = 500, activation = 'relu')(x)
    x=Dropout(rate = .5)(x)
    for i in range(input_shape[2]-1):
        flt = Flatten()(encoded[i+1])
        combined = concatenate([x, flt])
        x=Dense(units = 500+400*(i+1), activation = 'relu')(combined)
        x=Dropout(rate = .5)(x)
    forecast = Dense(units = 2025, activation = 'relu')(x)
    forecast = Reshape((45,45, 1))(forecast)
    
    

    ############
    # Decoding #
    ############

    # DeConv1
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(forecast)
    x = UpSampling2D((2, 2))(x)

    # DeConv2
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # Deconv3
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(x)
    
    return Model(inputs, decoded)
    
def getSameDayAutoEncoder(input_shape):

    input_img = Input(shape=input_shape)

    # Conv1 #
    x = Conv2D(filters = 16, kernel_size = (3, 3), activation='selu', padding='same')(input_img)
    x = AveragePooling2D(pool_size = (2, 2), padding='same')(x)
    # Conv2 #
    x = Conv2D(filters = 8, kernel_size = (3, 3), activation='selu', padding='same')(x)
    x = AveragePooling2D(pool_size = (2, 2), padding='same')(x)
    # Conv 3 #
    x = Conv2D(filters = 4, kernel_size = (3, 3), activation='selu', padding='same')(x)
    encoded = AveragePooling2D(pool_size = (2, 2), padding='same')(x)
    encoded = Flatten()(encoded)
    encoded = Dense(1, activation = 'linear')(encoded)
    encoded = BatchNormalization()(encoded)
    
    
    encoder = Model(input_img, encoded)
    
    # DeConv1
    decoder_input = Input(shape = (1))
    x = Dense(8100, activation = 'selu')(decoder_input)
    x = Reshape((45,45,4))(x)
    x = Conv2D(4, (3, 3), activation='selu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    # DeConv2
    x = Conv2D(8, (3, 3), activation='selu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    # Deconv3
    x = Conv2D(16, (3, 3), activation='selu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='linear', padding='same')(x)
    
    decoder = Model(decoder_input, decoded)
    
    
    autoencoder_input = Input(shape = input_shape)
    e = encoder(autoencoder_input)
    d = decoder(e)
    autoencoder = Model(autoencoder_input, d)
    
    
    return (autoencoder, encoder, decoder)
    
def getSequentialPreEncoded(input_shape):
    import tensorflow.keras as keras

    inputs = []
    encoded = []
    encoder = keras.models.load_model('encoder.model')
    for layer in encoder.layers:
        layer.trainable = False
    decoder = keras.models.load_model('decoder.model')
    for layer in decoder.layers:
        layer.trainable = False
    
    #encoders for each input
    for i in range(input_shape[2]):
        inputs.append(Input(shape=(input_shape[0], input_shape[1], 1)))
        encoded.append(encoder(inputs[i]))

    # c = []
    # for i in range(input_shape[2]):
        # c.append(Flatten()(encoded[i]))
    # x = concatenate([c[0], c[1], c[2], c[3], c[4]])
    
    #Forecast
    x = Flatten()(encoded[0])
    x=Dense(units = 500, activation = 'sigmoid')(x)
    x=Dropout(rate = .5)(x)
    for i in range(input_shape[2]-1):
        flt = Flatten()(encoded[i+1])
        combined = concatenate([x, flt])
        x=Dense(units = 500+400*(i+1), activation = 'sigmoid')(combined)
        x=Dropout(rate = .5)(x)
    x = Dense(units = 2025, activation = 'sigmoid')(x)
    x = Reshape((45,45, 1))(x)
    forecast = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    
    decoded = decoder(forecast)
    
    return Model(inputs, decoded)
    
def getModel(input_shape):
    import tensorflow.keras as keras
    import numpy as np
    

    ae_latent_len = 16
    
    encoder = keras.models.load_model('encoder.model')

    for layer in encoder.layers:
        layer.trainable = False
    decoder = keras.models.load_model('decoder.model')
    for layer in decoder.layers:
        layer.trainable = False
    
    #encoders for each input
    inp = Input(shape=input_shape)
    x = TimeDistributed(encoder)(inp)
    x = Reshape((5,ae_latent_len))(x)
    
    date_encoding = Input(shape = (5,13))
    x = Concatenate(axis=2)([x,date_encoding])
    
    
    
    
    output_len = 1
    regulariser = None
    hidden_shape = [10,20,30,40,50,60,70,80,90,100]
    ts_decoder_inputs = Input(shape=(output_len, 1))

    # ts_encoder_inputs = keras.layers.Input(shape=(5,8113))
    # # Create a list of RNN Cells, these are then concatenated into a single layer
    # # with the RNN layer.
    # ts_encoder_cells = []
    # for hidden_layers in hidden_shape:
        # ts_encoder_cells.append(keras.layers.GRUCell(hidden_layers,
                                                  # kernel_regularizer=regulariser,
                                                  # recurrent_regularizer=regulariser,
                                                  # bias_regularizer=regulariser))
                                                  
    # ts_encoder = keras.layers.RNN(ts_encoder_cells, return_state=True)
    
    # ts_encoder_outputs_and_states = ts_encoder(ts_encoder_inputs)
    # ts_encoder_states = ts_encoder_outputs_and_states[1:]
    
    
    
    

    # ts_decoder_cells = []
    # for hidden_layers in hidden_shape:
        # ts_decoder_cells.append(keras.layers.GRUCell(hidden_layers,
                                                  # kernel_regularizer=regulariser,
                                                  # recurrent_regularizer=regulariser,
                                                  # bias_regularizer=regulariser))
    # ts_decoder = keras.layers.RNN(ts_decoder_cells, return_sequences=True, return_state=True)
    
    # ts_decoder_outputs_and_states = ts_decoder(ts_decoder_inputs, initial_state=ts_encoder_states)
    # ts_decoder_outputs = ts_decoder_outputs_and_states[0]
    # # #ts_decoder_outputs = BatchNormalization()(ts_decoder_outputs)
    # # ts_decoder_dense = keras.layers.Dense(8100,
                                       # # activation='linear',
                                       # # kernel_regularizer=regulariser,
                                       # # bias_regularizer=regulariser)
    # # ts_decoder_outputs = ts_decoder_dense(ts_decoder_outputs)
    
    
    # o = ts_decoder_outputs_and_states#BatchNormalization()(ts_encoder_outputs_and_states)
    # tsed = keras.models.Model(inputs=[ts_encoder_inputs, ts_decoder_inputs], outputs=o[0])
    # print(tsed.output_shape)
    # print(tsed.summary())     
    # x = tsed([x,ts_decoder_inputs])     
    
    x = keras.layers.LSTM(80, return_sequences = True)(x)
    x = Flatten()(x)
    x = Concatenate()([x,date_encoding[:,4,:]])
    x = Dense(units = ae_latent_len, activation = 'linear')(x)
    x = Reshape((output_len, ae_latent_len))(x)
    
    decoded = TimeDistributed(decoder)(x)
    #decoded = BatchNormalization()(decoded)
    
    return Model([inp, ts_decoder_inputs, date_encoding], decoded)
   
def getModelWithVAE(input_shape, latent_size):
    import tensorflow.keras as keras
    import numpy as np
    import VAE
    vae = VAE.VAE()
    

    
    
    #pre VAE way - encoder = keras.models.load_model('encoder.model')

    encoder = vae.get_VAE((input_shape[1:]))[1]
    e = keras.models.load_model('encoder.model', compile = False, custom_objects={'vae_sampling': vae.vae_sampling})
    encoder.set_weights(e.get_weights())
    e = None
    
    for layer in encoder.layers:
        layer.trainable = False
    decoder = keras.models.load_model('decoder.model')
    for layer in decoder.layers:
        layer.trainable = False
    
    ae_latent_len = latent_size
    
    #encoders for each input
    inp = Input(shape=input_shape)
    x = TimeDistributed(encoder)(inp)
    x = Reshape((5,ae_latent_len))(x)
    
    date_encoding = Input(shape = (5,13))
    x = Concatenate(axis=2)([x,date_encoding])
    
    
    
    
    output_len = 1
    regulariser = None
    hidden_shape = [10,20,30,40,50,60,70,80,90,100]
    #ts_decoder_inputs = Input(shape=(output_len, 1))

    # ts_encoder_inputs = keras.layers.Input(shape=(5,8113))
    # # Create a list of RNN Cells, these are then concatenated into a single layer
    # # with the RNN layer.
    # ts_encoder_cells = []
    # for hidden_layers in hidden_shape:
        # ts_encoder_cells.append(keras.layers.GRUCell(hidden_layers,
                                                  # kernel_regularizer=regulariser,
                                                  # recurrent_regularizer=regulariser,
                                                  # bias_regularizer=regulariser))
                                                  
    # ts_encoder = keras.layers.RNN(ts_encoder_cells, return_state=True)
    
    # ts_encoder_outputs_and_states = ts_encoder(ts_encoder_inputs)
    # ts_encoder_states = ts_encoder_outputs_and_states[1:]
    
    
    
    

    # ts_decoder_cells = []
    # for hidden_layers in hidden_shape:
        # ts_decoder_cells.append(keras.layers.GRUCell(hidden_layers,
                                                  # kernel_regularizer=regulariser,
                                                  # recurrent_regularizer=regulariser,
                                                  # bias_regularizer=regulariser))
    # ts_decoder = keras.layers.RNN(ts_decoder_cells, return_sequences=True, return_state=True)
    
    # ts_decoder_outputs_and_states = ts_decoder(ts_decoder_inputs, initial_state=ts_encoder_states)
    # ts_decoder_outputs = ts_decoder_outputs_and_states[0]
    # # #ts_decoder_outputs = BatchNormalization()(ts_decoder_outputs)
    # # ts_decoder_dense = keras.layers.Dense(8100,
                                       # # activation='linear',
                                       # # kernel_regularizer=regulariser,
                                       # # bias_regularizer=regulariser)
    # # ts_decoder_outputs = ts_decoder_dense(ts_decoder_outputs)
    
    
    # o = ts_decoder_outputs_and_states#BatchNormalization()(ts_encoder_outputs_and_states)
    # tsed = keras.models.Model(inputs=[ts_encoder_inputs, ts_decoder_inputs], outputs=o[0])
    # print(tsed.output_shape)
    # print(tsed.summary())     
    # x = tsed([x,ts_decoder_inputs])     
    
    x = keras.layers.LSTM(1024, return_sequences = False)(x)
    x = Flatten()(x)
    x = Concatenate()([x,date_encoding[:,4,:]])
    x = Dense(units = ae_latent_len, activation = 'linear')(x)
    x = Reshape((output_len, ae_latent_len))(x)
    
    decoded = TimeDistributed(decoder)(x)
    #decoded = BatchNormalization()(decoded)
    
    return Model([inp, date_encoding], decoded)
   

