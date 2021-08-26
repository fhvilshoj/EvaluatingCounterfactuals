import os
import json
from tabulate import tabulate

from constants import METHOD_NAMES, METRIC_NAMES

format_fns = {
    'IM2': lambda mu, std, n: "%.2f (%.2f)" % (mu*100, std*100),
    'FID': lambda mu, std, n: "%.2f" % mu
}


def fn(args, cfg):
    out_path = os.path.join(cfg.get('checkpoints', 'output_dir'), 'results.json')

    if not os.path.exists(out_path):
        raise IOError("Result file does not exist, %s" % out_path)

    with open(out_path, 'r') as f:
        out_results = json.load(f) 

    # Print results in table.
    metric_names    = [m for m in METRIC_NAMES if m != 'LabelVariationScore']
    row_lookup      = {m: i+1 for i, m in enumerate(METHOD_NAMES)}

    cnt                 = 1
    col_lookup          = {}
    recorded_metrics    = []

    for metric in metric_names:
        if metric in out_results:
            col_lookup[metric] = cnt
            recorded_metrics.append(metric)
            cnt += 1

    results = [['Method \\ Metric'] + recorded_metrics]
    results.extend( [[m] + ['']*len(recorded_metrics)  for m in METHOD_NAMES] )

    for metric, col in col_lookup.items():
        for score in out_results[metric]:
            row         = row_lookup[score['method']]
            mu, ci, n   = score['scores']

            default_fmt         = lambda mu, std, n: '%.2f (%.2f)' % (mu, ci)
            results[row][col]   = format_fns.get(metric, default_fmt)(mu, ci, n)

    print("# # " * 10)
    print("# " + cfg.get('data', 'dataset').upper())
    print("# # " * 10)
    print(tabulate(results, headers='firstrow'))

    # Print LVS in separate table
    if 'LabelVariationScore' in out_results: 
        print("\n" + "- - " * 10 + 'LVS' + " - -" * 10 + "\n")
        lvs_scores = out_results['LabelVariationScore']
        
        rows = len(lvs_scores)
        cols = len(lvs_scores[0]['scores']) 

        if cols == 5: # Celeba
            label_names = ['Heavy_Makeup', 'High_Cheekbones', 'Attractive', 'Wearing_Lipstick', 'Smiling']
        else: # FakeMNIST
            label_names = ['FakeMNIST', 'MNIST']

        results = [['Method \\ Class Label'] + label_names]
        results.extend( [[m] + [''] * cols for m in METHOD_NAMES] )

        for lvs in lvs_scores:
            row = row_lookup[lvs['method']]
            for i, l in enumerate(label_names):
                col         = i+1
                mu, ci, n   = lvs['scores'][l]

                results[row][col] = "%.2f (%.2f)" % (mu, ci)

        print(tabulate(results, headers='firstrow'))

    
