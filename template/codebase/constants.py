from tensorflow.keras import optimizers

optimizers_map = {
    'SGD': optimizers.SGD,
    'RMSprop': optimizers.RMSprop,
    'Adam': optimizers.Adam,
    'Adadelta': optimizers.Adadelta,
    'Adagrad': optimizers.Adagrad,
    'Adamax': optimizers.Adamax,
    'Nadam': optimizers.Nadam,
    'Ftrl': optimizers.Ftrl,
}
