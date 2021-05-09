import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, Flatten, concatenate, Reshape, Dropout, LSTM, TimeDistributed, Concatenate, BatchNormalization, Lambda
from tensorflow.keras.models import Model



class GANModel(keras.Model):
    critic = None
    predictions = None
    critic_loss_scalar = 1
    regression_loss_scalar = 1
    critic_in = None
    
    def set_critic(self, critic):
        self.critic = critic
                
    def gan_loss(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred)
        diff = y_pred-y_true
        loss = keras.backend.square(diff)+keras.backend.abs(diff)*self.regression_loss_scalar#keras.backend.abs(diff+tf.math.divide(diff, 2))

        loss = keras.backend.min(loss, axis=0)+keras.backend.max(loss, axis=0)#keras.backend.sum(loss, axis=0)#    

        loss = loss+self.critic.predict_step(self.critic_in)*self.critic_loss_scalar
        return loss
           
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            self.critic_in = y_pred[1]
            loss = self.compiled_loss(y_true=y[0], y_pred=y_pred[0])
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        for m in self.metrics:
            print(m.name)
        return {m.name: m.result() for m in self.metrics}

def getGANModel(input_shape, latent_size):
    import tensorflow.keras as keras
    import numpy as np
    import VAE
    vae = VAE.VAE()
        
    encoder = vae.get_VAE((input_shape[1:]))[1]
    e = keras.models.load_model('encoder.model', compile = False, custom_objects={'vae_sampling': vae.vae_sampling})
    encoder.set_weights(e.get_weights())
    e = None
        
    for layer in encoder.layers:
        layer.trainable = False
    decoder = keras.models.load_model('decoder.model')
    # for layer in decoder.layers:
        # layer.trainable = False
        
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

        
    x = keras.layers.LSTM(1024, return_sequences = False)(x)
    x = Flatten()(x)
    x = Concatenate()([x,date_encoding[:,4,:]])
    latent_out = Dense(units = ae_latent_len, activation = 'linear')(x)
    x = Reshape((output_len, ae_latent_len))(latent_out)
        
    decoded = TimeDistributed(decoder)(x)
        
    return GANModel([inp, date_encoding], [decoded, latent_out])