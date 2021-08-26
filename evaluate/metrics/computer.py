import os
import numpy as np
from tqdm import tqdm

class Computer:
    def __init__(self, cfg, a, *args, **kwargs):
        self.cfg        = cfg
        self.args       = a

    def _iterate(self, elt_fn, aggr_fn, X, Xcf, Y, Yhat, Ycf, Yhatcf): 
        """
            elt_fn:     compute value to put in array
            aggr_fn:    aggregate values in array
        """
        out     = []
        total   = self.args.n_samples if self.args.n_samples > 0 else X.shape[0]

        n = 0
        for i, args in tqdm(enumerate(zip(X, Xcf, Y, Yhat, Ycf, Yhatcf)), total=total):
            if self.skip() and args[-1] == -1: continue # Skip failed 
            if self.skip() and args[1].mean() == self.cfg.getfloat('data', 'xmin'): continue
            else: n += 1

            # Compute score for sample i
            out.append(elt_fn(i, args))

            if self.args.n_samples > -1 and n >= self.args.n_samples: break

        return aggr_fn(out)

    def compute(self, *args):
        elt_fn  = lambda i, a: self.score(i, *a)
        aggr_fn = lambda s: (np.mean(s), 1.96*np.std(s)/np.sqrt(len(s)), len(s))
        return self._iterate(elt_fn, aggr_fn, *args)

    def __call__(self, *args, **kwargs): return self.compute(*args, **kwargs)
    
    # Interface.
    def skip(self):                                     return True # Skip unsuccessful CFs. 
    def score(self, i, x, xcf, y, yhat, ycf, yhatcf):   pass

        
