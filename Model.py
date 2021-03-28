from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, concatenate, Reshape, Dropout
from tensorflow.keras.models import Model

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
    decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
    
    return Model(inputs, decoded)