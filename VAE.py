import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, Lambda, Reshape
from tensorflow.keras.models import Model

class VAE:

    
    r_loss_factor = None
    mu = None
    log_var = None
    
    def __init__(self, r_loss_factor = 1):
        self.r_loss_factor = r_loss_factor
    
    def vae_r_loss(self, y_true, y_pred):
        r_loss = keras.backend.mean(keras.backend.square(y_true-y_pred), axis = [1,2,3])
        return self.r_loss_factor*r_loss
    
    def vae_kl_loss(self, y_true, y_pred):
        kl_loss = -.5*keras.backend.sum(1+self.log_var-keras.backend.square(self.mu)-keras.backend.exp(self.log_var), axis = 1)
        return kl_loss
    
    def vae_loss(self, y_true, y_pred):
        r_loss = self.vae_r_loss(y_true, y_pred)
        kl_loss = self.vae_kl_loss(y_true, y_pred)
        return r_loss+kl_loss
    
    
    def vae_sampling(self, args):
        mu, log_var = args
        self.mu = mu
        self.log_var = log_var
        epsilon = keras.backend.random_normal(shape = keras.backend.shape(mu), mean=0., stddev=1.)
        return mu+keras.backend.exp(log_var/2)*epsilon
        
       
    def get_VAE(self, input_shape):
        input_img = Input(shape=input_shape)
        latent_size = 32

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
        encoded = Dense(latent_size, activation = 'linear')(encoded)
        
        mu = Dense(latent_size, name = 'mu')(encoded)
        log_var = Dense(latent_size, name = 'log_var')(encoded)
        encoder_mu_log_var = Model(input_img, (mu, log_var))
        encoder_output = Lambda(self.vae_sampling, name='encoder_output')([mu, log_var])
        
        encoder = Model(input_img, encoder_output)
        
        # DeConv1
        decoder_input = Input(shape = (latent_size))
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