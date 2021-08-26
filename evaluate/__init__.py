import torch
import pickle
import os
from   tabulate import tabulate
import json
from constants import METHOD_NAMES

# Local imports
from .metrics import * 

def get_metrics(cfg, args):
    is_mnist = cfg.get('data', 'dataset').lower() == 'mnist'

    if args.all or args.im1  or args.im2: im1 = IM1(cfg, args) # avoid loading autoencoders multiple times

    metrics = []
    if args.all or args.im1:                        metrics.append(im1)
    if args.all or args.im2:                        metrics.append(IM2(cfg, args, aes=im1.aes))
    if args.all or args.tcv:                        metrics.append(TargetClassValidity(cfg, args))
    if args.all or args.en:                         metrics.append(ElasticNet(cfg, args))
    if args.all or args.oracle:                     metrics.append(Oracle(cfg, args))
    if (args.all and not is_mnist) or args.lvs:     metrics.append(LabelVariationScore(cfg, args))
    if args.all or args.fid:                        metrics.append(FID(cfg, args))

    return metrics

def fn(args, cfg, data):
    metrics         = get_metrics(cfg, args)

    res = {}
    for metric in metrics: 
        res[metric.name] = []
        for method in METHOD_NAMES:
            res[metric.name].append({'method': method, 'scores': metric(*data.organized_data(method))})

    # Store results
    out_results = {}

    out_path = os.path.join(cfg.get('checkpoints', 'output_dir'), 'results.json')
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            out_results = json.load(f) 
        
    for k, v in res.items():
        out_results[k] = v
            
    with open(out_path, 'w') as f:
        json.dump(out_results, f, sort_keys=True, indent=2)

    print("Done computing scores.\nStored results in %s" % out_path)

