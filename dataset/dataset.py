import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from .utils  import normalize, unnormalize

class Dataset():
    def __init__(self, cfg):
        self.dataset        = cfg.get('data', 'dataset')
        self.cfg            = cfg
        
        if 'mnist' in self.dataset.lower(): 
            self.dims       = (28, 28, 1)
            self.channels   = 1
            self.n_classes  = 10

        elif self.dataset.lower() == 'celeba':
            self.dims       = (64, 64, 3)
            self.channels   = 3
            self.n_classes  = 2

        L       = np.load('data/Real/%s.npz' % self.dataset)
        self.X  = normalize(L['x'], cfg).squeeze()
        self.Y  = L['y']

        if 'mnist' in self.dataset.lower(): 
            self.Y = to_categorical(self.Y)

        def load_from_method(method): 
            L   = np.load('data/%s/%s.npz' % (method, self.dataset))
            cfs = normalize(L['cfs'], cfg).squeeze()
            return cfs, L['orig_pred'], L['q'], L['cf_pred']
        
        self.GB     = list( load_from_method('GB')  )
        self.GL     = list( load_from_method('GL')  )
        self.GEN    = list( load_from_method('GEN') )

    def unnormalize(self, x): 
        return unnormalize(x, self.cfg)

    def organized_data(self, method):
        # Returns data in proper order to be used with computers
        try: 
            D = getattr(self, method)
        except AttributeError:
            raise AttributeError("%s is not a valid method" % method)

        #         X    Xcf    Y     Yhat   q    Yhatcf
        return self.X, D[0], self.Y, D[1], D[2], D[3]


