import numpy as np

def reweight_events(original_weights, targeted_weights=None):
    pass

class LimitedDict(dict):

    _struct = {}

    def __init__(self):
        for k, v in LimitedDict._struct:
            self[k] = v

    def __setitem__(self, k, v):
        if k not in LimitedDict._struct.keys():
            raise KeyError()
        elif not isinstance(v, _struct[k]):
            raise TypeError()
        else:
            self[k] = v

class SetDict(LimitedDict):

    _struct = {
        'x_raw': int,
        'x': int,
        'y_true_raw': int,
        'y_true': int,
        'y_pred': int,
        'weights_raw': int,
        'weights': int,
    }

class DataDict(LimitedDict):

    _struct = {
        'train': SetDict,
        'val': SetDict,
        'test': SetDict,
    }

if __name__ == '__main__':
    # x = DataDict()
    # x['train'] = SetDict()

    x = np.zeros(shape=(1000000, 50,), dtype=float)
    print(x.dtype)
    np.save('test.npy', x)
