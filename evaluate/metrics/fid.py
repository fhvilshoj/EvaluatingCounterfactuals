import os
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import load_model
from tqdm import tqdm
from numpy.linalg import norm
from scipy.linalg import sqrtm

from .computer import Computer
from .utils import fix_shape, get_model_dir


class FID(Computer):
    def __init__(self, *args, **kwargs):
        super(FID, self).__init__(*args, **kwargs)

        self.enc        = self.get_encoder_model()

        self.name       = type(self).__name__ 
        self.desc       = """FID:
              
        FID = ||\\mu_1 - \\mu_2||^2_2 + tr[\\Sigma_1 +  \\Sigma_2 - 2tr[\\sqrt{\\Sigma_1\\Sigma_2}]
        
        """

    def get_encoder_model(self):
        dataset = self.cfg.get('data', 'dataset').lower().replace("_", "")
        if dataset == 'celeba':
            fname = 'independent_makeup.h5'
        else:
            fname = 'independent.h5'

        pth         = get_model_dir(self.cfg, fname)
        full_model  =  load_model(pth)

        model       = Sequential(full_model.layers[:-3]) # Remove final prediction layer (and a dropout layer).
        model.add( Dense(256, activation=None) ) # Add last dense layer without ReLU
        model.layers[-1].set_weights(full_model.layers[-3].get_weights()) 

        return model

    def compute(self, X, Xcf, Y, Yhat, Ycf, Yhatcf):
        real_encodings = []
        fake_encodings = []

        # Batch samples to speed up computations.
        total   = self.args.n_samples if self.args.n_samples > 0 else X.shape[0]
        bs      = 256
        steps   = total // bs + int(total % bs != 0)

        Px      = []
        Pcf     = []
        for s in range(steps):
            x   = fix_shape(X[s*bs:(s+1)*bs])
            xcf = fix_shape(Xcf[s*bs:(s+1)*bs])
            
            Px.append(self.enc.predict(x).reshape(-1, 256)) 
            Pcf.append(self.enc.predict(xcf).reshape(-1, 256))
        Px  = np.concatenate(Px, 0)
        Pcf = np.concatenate(Pcf, 0)

        n = 0
        for i, args in tqdm(enumerate(zip(X, Xcf, Y, Yhat, Ycf, Yhatcf)), total=total):
            if self.skip() and args[-1] == -1: continue # Skip failed 
            if self.skip() and args[1].mean() == self.cfg.getfloat('data', 'xmin'): continue
            else: n += 1

            real_encodings.append( Px[i] )
            fake_encodings.append( Pcf[i] )

            if self.args.n_samples > -1 and n >= self.args.n_samples: break

        E_real  = np.stack(real_encodings)
        E_fake  = np.stack(fake_encodings)

        mu_real = np.mean(E_real, axis=0)
        mu_fake = np.mean(E_fake, axis=0)

        S_real  = np.cov(E_real, rowvar=False)
        S_fake  = np.cov(E_fake, rowvar=False)

        Sqrt    = np.real( sqrtm( S_real @ S_fake ) )
        res     = norm( mu_real - mu_fake )**2 + np.trace(S_real) + np.trace(S_fake) - 2*np.trace( Sqrt )

        return res, -1, E_real.shape[0]
