from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Adagrad

from constants import INPUT_SHAPE, NUM_CLASSES

AVAILABLE_OPTIMIZERS = {
    'sgd': SGD,
    'adagrad': Adagrad,
    'adadelta': Adadelta,
    'rmsprop': RMSprop,
    'adam': Adam,
}


def get_model(model_name, dropout_rate, learning_rate, optimizer, augmentation):
    """
    Retrieve a Sequential model based on model_name

    Parameters:
    - model_name (str): Name of the model, must be a key in MODELS_CONFIG

    Returns:
    - keras.Sequential: Sequential model

    Example:
    - model = get_model('SimpleCNN')

    """
    MODELS_CONFIG = {
        # TODO: add more models DDN1, DNN2, CNN1, CNN2, CNN3
        'DNN1': Sequential([
            Flatten(input_shape=INPUT_SHAPE),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),
        ]),
        'CNN1': Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=128, activation='relu'),
            Dropout(rate=dropout_rate),
            Dense(NUM_CLASSES, activation='softmax'),

            # TODO: BatchNormalization
            # TODO: augmentation
        ]),
        'CNN3': [...],
        'CNN8': [...],
        '...': [...],
    }
    assert model_name in MODELS_CONFIG.keys(), f"Unknown model name: {model_name}, choose one of {MODELS_CONFIG.keys()}"
    model = MODELS_CONFIG[model_name]
    assert optimizer in AVAILABLE_OPTIMIZERS.keys(), f"Unknown optimizer: {optimizer}, choose one of {AVAILABLE_OPTIMIZERS.keys()}"

    optimizer_obj = AVAILABLE_OPTIMIZERS[optimizer](learning_rate=learning_rate)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer_obj,
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    get_model(model_name='SimpleCNN', dropout_rate=0.02, learning_rate=0.01, optimizer='adam', augmentation=False)
