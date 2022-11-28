import pandas as pd
import pickle as pkl

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

class AdditionalValidationSets(Callback):

    def __init__(self, datasets, convert_to_tensor=True, evaluate_kwargs={}, verbose=True, history_filepath=None):
        '''
        # :param list datasets (list of 2-tuples)
        :param dict datasets
        :param dict evaluate_kwargs: arguments to model.evaluate()
        '''
        super(AdditionalValidationSets, self).__init__()
        self.datasets = datasets
        if convert_to_tensor:
            # https://stackoverflow.com/questions/64199384
            for dataset_k, dataset_v in self.datasets.items():
                self.datasets[dataset_k]['x'] = tf.convert_to_tensor(dataset_v['x'], dtype=tf.float32)
                self.datasets[dataset_k]['y_true'] = tf.convert_to_tensor(dataset_v['y_true'], dtype=tf.float32)
        self.evaluate_kwargs = evaluate_kwargs
        self.history = {}
        self.verbose = verbose
        self.history_filepath = history_filepath

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        # logs = logs or {}
        self.epoch.append(epoch)
        self.history.setdefault('epoch', []).append(epoch)
        # for k, v in logs.items():
        #     self.history.setdefault(k, []).append(v)
        for dataset_k, dataset_v in self.datasets.items():
            if self.verbose: print('Start evaluation of dataset \"{}\"'.format(dataset_k))
            results = self.model.evaluate(
                x=dataset_v['x'],
                y=dataset_v['y_true'],
                # sample_weight=dataset_v['weights'],
                sample_weight=(pd.Series(dataset_v['weights']).to_frame('weights') if dataset_v.get('weights') is not None else None),
                verbose=0, # no progress bar
                **self.evaluate_kwargs,
            )
            for i, result in enumerate(results):
                k = '_'.join([dataset_k, self.model.metrics_names[i]])
                if self.verbose: print('{}:'.format(k), result)
                self.history.setdefault(k, []).append(result)
        # save history after each epoch
        if self.history_filepath is not None:
            with open(self.history_filepath, 'wb') as f:
                pkl.dump(self.history, f)
