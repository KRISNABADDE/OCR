from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Reshape, Dense, Bidirectional, LSTM, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class OCRModel:
    def __init__(self, img_w, img_h, num_classes, max_length, drop_out_rate=0.25) -> None:
        self.img_w = img_w
        self.img_h = img_h
        self.num_classes = num_classes
        self.max_length = max_length
        self.drop_out_rate = drop_out_rate

    def model_building(self):
        input_shape = (1, self.img_w, self.img_h) if K.image_data_format() == 'channels_first' else (self.img_w, self.img_h, 1)
        
        model_input = Input(shape=input_shape, name='img_input', dtype='float32')

        # Convolutional layers
        x = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(model_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='max1')(x)

        x = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='max2')(x)

        x = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(x)
        x = Dropout(self.drop_out_rate)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1, 2), name='max3')(x)

        x = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv6')(x)
        x = Dropout(self.drop_out_rate)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1, 2), name='max4')(x)

        x = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='conv7')(x)
        x = Dropout(self.drop_out_rate)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Calculate the reshape target shape
        output_shape = K.int_shape(x)
        reshaped_shape = (output_shape[1] * output_shape[2], output_shape[3])

        # CNN to RNN
        x = Reshape(target_shape=reshaped_shape, name='reshape')(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(x)

        # RNN layers
        x = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(x)
        x = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat')(x)

        # Output layer
        x = Dense(self.num_classes, kernel_initializer='he_normal', name='dense2')(x)
        y_pred = Activation('softmax', name='softmax_layer')(x)

        return Model(inputs=model_input, outputs=y_pred)
    
    def training(self):
        self.model = self.model_building()
        labels = Input(name='ground_truth_labels', shape=[self.max_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        y_pred = self.model.output

        def ctc_loss_function(args):
            y_pred, labels, input_length, label_length = args
            return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

        loss_out = Lambda(ctc_loss_function, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
        train_model = Model(inputs=[self.model.input, labels, input_length, label_length], outputs=loss_out)
        
        return train_model
