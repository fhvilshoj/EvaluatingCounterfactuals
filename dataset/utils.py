import configparser
import numpy as np

def get_base_pth():
    cfg = configparser.ConfigParser()
    cfg.read('config.ini') # Default configurations 
    return cfg.get('data', 'cf_output_dir')

def normalize_fn(cfg):
    xmin = cfg.getfloat('data', 'xmin')
    xmax = cfg.getfloat('data', 'xmax')

    def _normalize(X): 
        X = np.reshape(X, X.shape + (1,))

        # Normalize X 
        X = X - X.min()
        X = X / X.max()
        # Rescale
        X = X * (xmax - xmin) + xmin
        
        return X

    return _normalize

def normalize(X, cfg):
    return normalize_fn(cfg)(X)

def unnormalize_fn(cfg):
    xmin       = cfg.getfloat('data', 'xmin')
    xmax       = cfg.getfloat('data', 'xmax') 

    def _unnormalize(X): 
        X  = np.clip(X, xmin, xmax)
        X  = X - xmin
        X  = X / (xmax - xmin)
        X *= 255
        return X.astype(np.uint8)

    return _unnormalize

def unnormalize(X, cfg): 
    return unnormalize_fn(cfg)(X)
