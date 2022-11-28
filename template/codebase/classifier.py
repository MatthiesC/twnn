import os
import sys
import json
import pickle
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow import keras

import uproot
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from .callbacks import AdditionalValidationSets

class SimpleEventClassifier:

    def __init__(self, config):
        print('Hello World from', self.__class__.__name__)


        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


        self.raw_data_dir = '/beegfs/desy/user/matthies/WorkingArea/twnn/data/raw'
        self.config = config
        self.workdir = '.' # FIXME
        self.datadir = os.path.join(self.workdir, 'data')
        os.makedirs(self.datadir, exist_ok=True)

        self.filepaths = {}
        self.filepaths['model_arch'] = os.path.join(self.workdir, 'model_arch.json')
        self.filepaths['model_as_png'] = os.path.join(self.workdir, 'model_arch.png')
        self.filepaths['checkpoints'] = os.path.join(self.workdir, 'checkpoints', 'checkpoint_{epoch}.h5')
        self.filepaths['history'] = os.path.join(self.workdir, 'model_history.pkl')
        self.filepaths['customHistory'] = os.path.join(self.workdir, 'model_customHistory.pkl')

        print('Configuration:\n', json.dumps(config, indent=4))
        self.n_classes = len(self.config['classes'])
        if self.n_classes == 2:
            self.binary = True
            print('--> Configured as binary classifier')
            for class_k in self.config['classes'].keys():
                if class_k not in ['sig', 'bkg']:
                    sys.exit('Two classes found in config. This implies a binary classifier but the class names are not \"sig\"/\"bkg\". Abort')
        elif self.n_classes > 2:
            self.binary = False
            print('--> Configured as multi-classifier')
            class_index = 0
            self.class_to_index = {}
            # self.index_to_class = {}
            for class_k, class_v in self.config['classes'].items():
                if class_k == 'all':
                    sys.exit('Class cannot be named \"all\". This name is reserved for internal stuff')
                class_v['index'] = class_index
                print('Class \"'+class_k+'\" gets index '+str(class_index))
                self.class_to_index[class_k] = class_index
                # self.index_to_class[class_index] = class_k
                class_index += 1
            self.index_to_class = {class_v['index']: class_k for class_k, class_v in self.config['classes'].items()}
            self.filepaths['class_to_index'] = os.path.join(self.workdir, 'class_to_index.pkl')
            with open(self.filepaths['class_to_index'], 'wb') as f:
                pickle.dump(self.class_to_index, f)
            self.filepaths['index_to_class'] = os.path.join(self.workdir, 'index_to_class.pkl')
            with open(self.filepaths['index_to_class'], 'wb') as f:
                pickle.dump(self.index_to_class, f)
        else:
            raise ValueError('Number of classes <= 1. Abort')
        ### Setup filepaths for raw pandas data
        self.datadir_raw_pandas = os.path.join(self.datadir, 'raw_pandas')
        os.makedirs(self.datadir_raw_pandas, exist_ok=True)
        # self.datadir_raw_pandas = self.raw_data_dir
        for class_k, class_v in self.config['classes'].items():
            for subclass_k, subclass_v in class_v['subclasses'].items():
                for data_type in ['x', 'weights']:
                    filename = 'class_'+class_k+'__subclass_'+subclass_k+'__'+data_type+'.pandas_pkl'
                    filepath = os.path.join(self.datadir_raw_pandas, filename)
                    self.filepaths.setdefault('raw_pandas', {}).setdefault(data_type, {}).setdefault(class_k, {})[subclass_k] = filepath
        self.history = {}
        self.inputs = {}

    def get_input_keys(self, load_from_scratch=True):
        self.inputs = {}
        if load_from_scratch:
            # take "file" from first subclass of first class; we assume that AnalysisTree structure is consistent with all other root files
            rootFileName = next(iter(next(iter(self.config['classes'].values()))['subclasses'].values()))['file']
            rootFile = uproot.open(
                os.path.join(self.raw_data_dir, rootFileName)+':AnalysisTree',
            )
        else:
            pass # FIXME
        for input_postfix in self.config['inputs']:
            input_wildcard = self.config['input_prefix']+input_postfix
            if load_from_scratch:
                inputs = rootFile.keys(
                    filter_name=input_wildcard,
                )
            else:
                pass # FIXME
            for input in inputs:
                self.inputs[input] = {}
        return self.inputs.keys()

    def get_class_mask(self, class_k, y_true):
        if self.binary:
            class_mask = y_true == (1 if class_k == 'sig' else 0)
        else:
            class_mask = y_true[:,self.class_to_index[class_k]].flatten() != 0
        return class_mask

    def prepare_data(self, load_from_scratch=True):
        print('Preparing data')
        self.data = {}
        for dataset_name in ['train', 'val', 'test']:
            self.data[dataset_name] = {
                'x': None,
                'y_true': None,
                'y_pred': None, # not filled in this function
                'weights': None,
            }

        for class_k, class_v in self.config['classes'].items():
            print('Reading data for class', class_k)
            class_x = None # shpae depends on number of inputs that is not yet known
            class_weights = np.array([], dtype=np.float32)
            for subclass_k, subclass_v in class_v['subclasses'].items():
                print('Reading data for subclass', subclass_k)
                start_time = timer()
                # rootFile = uproot.open(
                #     os.path.join(self.raw_data_dir, subclass_v['file'])+':AnalysisTree',
                # )
                # self.inputs = {}
                # for input_postfix in self.config['inputs']:
                #     input_wildcard = self.config['input_prefix']+input_postfix
                #     inputs = rootFile.keys(
                #         filter_name=input_wildcard,
                #     )
                #     for input in inputs:
                #         self.inputs[input] = {}
                if class_x is None:
                    class_x = np.array([], dtype=np.float32).reshape(0, len(self.inputs))
                #################
                if load_from_scratch:
                    cuts = []
                    if self.config.get('cut') is not None:
                        cuts.append(self.config['cut'])
                    if subclass_v.get('cut') is not None:
                        cuts.append(subclass_v['cut'])
                    if len(cuts):
                        cut = ' & '.join(['('+c+')' for c in cuts])
                    else:
                        cut = None
                    aliases = {
                        'theWeight': subclass_v.get('theWeight', 'weight')
                    }
                    rootFile = uproot.open(
                        os.path.join(self.raw_data_dir, subclass_v['file'])+':AnalysisTree',
                    )
                    subclass_x = rootFile.arrays(
                        library='pd',
                        expressions=self.inputs.keys(),
                        cut=cut,
                    )#.to_numpy()
                    subclass_x.to_pickle(self.filepaths['raw_pandas']['x'][class_k][subclass_k])
                    subclass_weights = rootFile.arrays(
                        library='pd',
                        expressions=['theWeight'],
                        aliases=aliases,
                        cut=cut,
                    )#.to_numpy().flatten()
                    subclass_weights.to_pickle(self.filepaths['raw_pandas']['weights'][class_k][subclass_k])
                else:
                    subclass_x = pd.read_pickle(self.filepaths['raw_pandas']['x'][class_k][subclass_k])
                    subclass_weights = pd.read_pickle(self.filepaths['raw_pandas']['weights'][class_k][subclass_k])
                subclass_x = subclass_x.to_numpy()
                subclass_weights = subclass_weights.to_numpy().flatten()
                if len(subclass_x[0]) != len(self.inputs):
                    if load_from_scratch:
                        sys.exit('Number of inputs does not match number of columns in x. Something went seriously wrong')
                    else:
                        sys.exit('Number of inputs changed. Cannot reuse pandas from disk')
                # shuffle the events
                subclass_x, subclass_weights = shuffle(subclass_x, subclass_weights, random_state=0)
                # take only a fraction of the events (useful if MC sample is huge but its size does not benefit training)
                fraction = subclass_v.get('fraction')
                if fraction:
                     # shuffling before this slicing is mandatory!
                    n_events = len(subclass_x)
                    subclass_x = subclass_x[:int(fraction*n_events)]
                    subclass_weights = subclass_weights[:int(fraction*n_events)]
                    subclass_weights = subclass_weights / fraction # need to rescale the weights with the fraction
                # Now remove NaNs:
                nan_mask_x = ~np.isnan(subclass_x).any(axis=1)
                nan_mask_weights = ~np.isnan(subclass_weights)
                nan_mask = nan_mask_x & nan_mask_weights
                subclass_x = subclass_x[nan_mask]
                subclass_weights = subclass_weights[nan_mask]
                print('Found and removed {} NaN-affected events out of {} events'.format(len(nan_mask[nan_mask==False]), len(nan_mask)))
                class_x = np.append(class_x, subclass_x, axis=0)
                class_weights = np.append(class_weights, subclass_weights, axis=0)
                print('Loading time: {:.2f} sec'.format(timer() - start_time))
            if self.binary:
                if class_k == 'sig':
                    class_y_true = np.ones(len(class_x), dtype=np.float32)
                elif class_k == 'bkg':
                    class_y_true = np.zeros(len(class_x), dtype=np.float32)
            else:
                y_true_single = np.zeros(self.n_classes, dtype=np.float32)
                y_true_single[class_v['index']] = 1.
                class_y_true = np.full((len(class_x), len(y_true_single)), y_true_single)

            # shuffle the subclasses
            if self.binary:
                if class_k == 'sig':
                    random_state = 1
                else:
                    random_state = 0
            else:
                random_state = class_v['index']
            class_x, class_y_true, class_weights = shuffle(class_x, class_y_true, class_weights, random_state=random_state)
            class_v['x'] = class_x
            class_v['y_true'] = class_y_true
            if self.config.get('normalize_sample_weights'):
                class_v['weights'] = class_weights / np.sum(class_weights) # each class gets total weight of 1 --> no underrepresented classes

        for class_k, class_v in self.config['classes'].items():
            length_class = len(class_v['weights'])

            slice_indices = [
                0,
                int(length_class*self.config['fraction_train']), # end of train
                int(length_class*0.5*(1.+self.config['fraction_train'])), # end of val
                length_class, # end of test
            ]

            for dataset_i, dataset_name in enumerate(['train', 'val', 'test']):
                if self.data[dataset_name].get('weights') is None:
                    self.data[dataset_name]['weights'] = np.array([], dtype=np.float32)
                if self.data[dataset_name].get('x') is None:
                    self.data[dataset_name]['x'] = np.array([], dtype=np.float32).reshape(0, len(self.inputs))
                if self.data[dataset_name].get('y_true') is None:
                    if self.binary:
                        self.data[dataset_name]['y_true'] = np.array([], dtype=np.float32)
                    else:
                        self.data[dataset_name]['y_true'] = np.array([], dtype=np.float32).reshape(0, self.n_classes)
                self.data[dataset_name]['weights'] = np.append(self.data[dataset_name]['weights'], class_v['weights'][slice_indices[dataset_i]:slice_indices[dataset_i+1]], axis=0)
                self.data[dataset_name]['x'] = np.append(self.data[dataset_name]['x'], class_v['x'][slice_indices[dataset_i]:slice_indices[dataset_i+1]], axis=0)
                self.data[dataset_name]['y_true'] = np.append(self.data[dataset_name]['y_true'], class_v['y_true'][slice_indices[dataset_i]:slice_indices[dataset_i+1]], axis=0)

        # get input normalization scheme from train data
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.data['train']['x'])
        for input_i, input_v in enumerate(self.inputs.values()):
            input_v['scale'] = self.scaler.scale_[input_i]
            input_v['min'] = self.scaler.min_[input_i]
            input_v['offset'] = self.scaler.min_[input_i] / self.scaler.scale_[input_i] # used for lwtnn

        for dataset_i, dataset_name in enumerate(['train', 'val', 'test']):
            # apply input normalization to train, val, and test data
            self.data[dataset_name]['x'] = self.scaler.transform(self.data[dataset_name]['x'])
            # shuffle now the classes
            self.data[dataset_name]['weights'], self.data[dataset_name]['x'], self.data[dataset_name]['y_true'] = shuffle(self.data[dataset_name]['weights'], self.data[dataset_name]['x'], self.data[dataset_name]['y_true'], random_state=dataset_i)
            # save the data to disk
            for data_type in ['x', 'y_true', 'weights']:
                filepath = os.path.join(self.datadir, dataset_name+'__'+data_type+'.npy')
                self.filepaths.setdefault('data', {}).setdefault(dataset_name, {}).setdefault(data_type, {})['all'] = filepath
                np.save(filepath, self.data[dataset_name][data_type])
            # save the data to disk also separated by classes
            for class_k, class_v in self.config['classes'].items():
                class_mask = self.get_class_mask(class_k, self.data[dataset_name]['y_true'])
                for data_type in ['x', 'weights']:
                    masked = self.data[dataset_name][data_type][class_mask]
                    filepath = os.path.join(self.datadir, dataset_name+'__'+data_type+'__'+class_k+'.npy')
                    self.filepaths.setdefault('data', {}).setdefault(dataset_name, {}).setdefault(data_type, {})[class_k] = filepath
                    np.save(filepath, masked)

        print('Save input information')
        self.filepaths['inputs'] = os.path.join(self.workdir, 'inputs.pkl')
        with open(self.filepaths['inputs'], 'wb') as f:
            pickle.dump(self.inputs, f)

        print('Write lwtnn-compatible \"variables.json\"')
        self.filepaths['lwtnn_variables'] = os.path.join(self.workdir, 'variables.json')
        lwtnn_inputs = []
        for input_k, input_v in self.inputs.items():
            lwtnn_inputs.append({'name': input_k, 'scale': float(input_v['scale']), 'offset': float(input_v['offset'])}) # need to undo np.float32 which is not JSON serializable
        if self.binary:
            lwtnn_class_labels = ['binary_score']
        else:
            lwtnn_class_labels = []
            for class_k in self.config['classes'].keys():
                class_label = '_'.join(['node', str(self.class_to_index[class_k]), class_k])
                lwtnn_class_labels.append(class_label)
        with open(self.filepaths['lwtnn_variables'], 'w', encoding='utf-8') as f:
            lwtnn_variables = {
                'inputs': lwtnn_inputs,
                'class_labels': lwtnn_class_labels,
            }
            json.dump(lwtnn_variables, f, indent=4)

        print('Data prepared')

    def build_model_from_scratch(self):
        print('Building architecture')
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.InputLayer(input_shape=(len(self.inputs))))
        input_dropout = self.config.get('input_dropout')
        if input_dropout is not None:
            self.model.add(keras.layers.Dropout(input_dropout))
        for hidden_layer in self.config['hidden_layers']:
            self.model.add(keras.layers.Dense(hidden_layer['units']))
            if self.config.get('batch_norm'):
                self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Activation(hidden_layer['activation']))
            if hidden_layer.get('dropout'):
                self.model.add(keras.layers.Dropout(hidden_layer['dropout']))
        if self.binary:
            self.model.add(keras.layers.Dense(1)) # not self.n_classes; we have two classes for binary classifier but only want one output node!
            self.model.add(keras.layers.Activation('sigmoid'))
        else:
            self.model.add(keras.layers.Dense(self.n_classes))
            self.model.add(keras.layers.Activation('softmax'))
        # self.filepaths['model_arch'] = os.path.join(self.workdir, 'model_arch.json')
        with open(self.filepaths['model_arch'], 'w') as f:
            f.write(self.model.to_json())
        # self.filepaths['model_as_png'] = os.path.join(self.workdir, 'model_arch.png')
        keras.utils.plot_model(self.model, to_file=self.filepaths['model_as_png'])

    def load_model_from_checkpoint(self, epoch):
        '''
        :param int epoch
        '''
        print('Load model from disk')
        with open(self.filepaths['model_arch'], 'r') as json_file:
            self.model = keras.models.model_from_json(json_file.read())
        filepath_weights = self.filepaths['checkpoints'].format(epoch=epoch)
        print('Checkpoint:', filepath_weights)
        self.model.load_weights(filepath_weights)

    def init_model(self, from_checkpoint=None, compile=True):
        if from_checkpoint is not None:
            self.load_model_from_checkpoint(epoch=from_checkpoint)
        else:
            self.build_model_from_scratch()
        if compile:
            self.compile()

    def compile(self):
        print('Compiling model')
        lr_schedule = getattr(keras.optimizers.schedules, self.config['lr_schedule']['name'])(**self.config['lr_schedule']['kwargs'])
        # initial_learning_rate = 0.0001
        # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate,
        #     decay_steps = 10000,
        #     decay_rate = 0.9,
        #     staircase = False,
        # )
        optimizer = keras.optimizers.Adam(
            learning_rate = lr_schedule,
        )
        if self.binary:
            metrics = [keras.metrics.BinaryAccuracy()]
            loss = keras.losses.BinaryCrossentropy(from_logits=True)
        else:
            metrics = [keras.metrics.CategoricalAccuracy()]
            loss = keras.losses.CategoricalCrossentropy(from_logits=False)
        self.model.compile(
            optimizer=optimizer,
            weighted_metrics=metrics, # needs to be weighted_metrics, not just metrics, because we use sample_weight
            loss=loss,
        )

    def setup_callbacks(self, callbacks=None):
        '''
        :param list callbacks: user-defined list of additional Callback instances for the model
        '''
        print('Setting up callbacks')
        self.callbacks = callbacks or []
        #__________________________________________
        os.makedirs(os.path.dirname(self.filepaths['checkpoints']), exist_ok=True)
        self.callbacks.append(
            keras.callbacks.ModelCheckpoint(
                self.filepaths['checkpoints'],
                verbose=1,
                save_weights_only=True,
                save_freq='epoch',
            )
        )
        #__________________________________________
        self.history['custom'] = AdditionalValidationSets(
            datasets=dict([
                ('train', self.data['train']),
                ('val', self.data['val']),
            ]),
            evaluate_kwargs={'batch_size': self.config['batch_size']},
            history_filepath=self.filepaths['customHistory'],
        )
        self.callbacks.append(self.history['custom'])
        #__________________________________________
        if self.config.get('early_stopping'):
            self.callbacks.append(
                keras.callbacks.EarlyStopping(
                    **self.config['early_stopping']
                )
            )

    def fit(self):
        print('Fitting!')
        # with tf.device('/gpu:0'):
        self.history['default'] = self.model.fit(
            x=self.data['train']['x'],
            y=self.data['train']['y_true'],
            sample_weight=pd.Series(self.data['train']['weights']).to_frame('weights'), # https://github.com/keras-team/keras/issues/14877#issuecomment-1254581771 and https://stackoverflow.com/questions/63158424
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            shuffle=True,
            validation_data=(
                self.data['val']['x'],
                self.data['val']['y_true'],
                self.data['val']['weights'],
            ),
            callbacks=self.callbacks,
            # verbose=0, # no progress bar
        )
        # self.model_path = os.path.join(self.workdir, 'model.h5')
        # self.model.save(self.model_path)
        # self.model_weights_path = os.path.join(self.workdir, 'model_weights.h5')
        # self.model.save_weights(self.model_weights_path)
        history = {}
        for history_k, history_v in self.history.items():
            history[history_k] = history_v.history
        print('Save history')
        with open(self.filepaths['history'], 'wb') as f:
            pickle.dump(history, f)

    def resume(self):
        pass

    # def get_best_model(self, compile=False):
    #     with open(self.filepaths['model_arch'], 'r') as f:
    #         self.model = keras.models.model_from_json(f.read())
    #
    #     # FIXME
    #     # decide which model is the best
    #     self.best = {}
    #
    #     self.filepaths['best_model'] = os.path.join(self.workdir, 'best_model.h5')
    #     self.model.load_weights(self.filepaths['best_model'])
    #     if compile:
    #         self.compile()

    def predict(self):
        for dataset_name in ['train', 'val', 'test']:
            print('Predict labels for \"', dataset_name, '\" dataset')
            # with tf.device('/gpu:0'):
            # convert to tf tensor before passing to predict to avoid numpy-related memory leak, see https://stackoverflow.com/questions/64199384
            dataset = tf.convert_to_tensor(self.data[dataset_name]['x'], dtype=tf.float32)
            self.data[dataset_name]['y_pred'] = self.model.predict(
                x=dataset,
                # verbose=0, # no progress bar
                batch_size=self.config['batch_size'],
            )
            # save predictions to disk
            filepath = os.path.join(self.datadir, dataset_name+'__y_pred.npy')
            self.filepaths.setdefault('data', {}).setdefault(dataset_name, {}).setdefault('y_pred', {})['all'] = filepath
            np.save(filepath, self.data[dataset_name]['y_pred'])
            # save the data to disk also separated by classes
            for class_k, class_v in self.config['classes'].items():
                class_mask = self.get_class_mask(class_k, self.data[dataset_name]['y_true'])
                masked = self.data[dataset_name]['y_pred'][class_mask]
                filepath = os.path.join(self.datadir, dataset_name+'__y_pred__'+class_k+'.npy')
                self.filepaths.setdefault('data', {}).setdefault(dataset_name, {}).setdefault('y_pred', {})[class_k] = filepath
                np.save(filepath, masked)


# if __name__=='__main__':
#     x = SimpleEventClassifier()
