import os
from tensorflow.keras.models import load_model

def get_model_dir(cfg, suffix=''):
    if cfg.has_option('checkpoints', 'eval_model_dir'):
        d = cfg.get('checkpoints', 'eval_model_dir')
    else: 
        d = os.path.join('./ckpts', cfg.get('data', 'dataset').lower().replace("_", ""))
    
    if len(suffix) > 0:
        d = os.path.join(d, suffix)
    return d

def load_aes(cfg):
    ckpt_dir = get_model_dir(cfg, suffix='aes')
    rng = range(2) if 'celeba' in cfg.get('data', 'dataset') else range(10)
    return [load_model(os.path.join(ckpt_dir, 'ae_%s.h5'%s)) for s in list(map(str,rng)) + ['all']]

def fix_shape(x): 
    if 28 in x.shape:   d, c = 28, 1
    else:               d, c = 64, 3
    return x.reshape(-1, d, d, c)

