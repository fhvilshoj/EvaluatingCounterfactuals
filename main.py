import os
import argparse

from dataset import Dataset

# Functions
import evaluate

def check_cfg(cfg): assert cfg is not None, "To use the model or the data, you must specify config:\n > python main.py [command] -c [int | output_dir_name]"

def main(): 
    parser      = argparse.ArgumentParser()
    parser.add_argument('--config',     '-c', type=str,     nargs="*",          help="Config file to use when loading/training/evaluating model")
    parser.add_argument('--query',      '-q', type=str,                         help='Query string to choose multiple configs.')
    parser.add_argument('--output_dir', '-o', type=str, default='./output',     help='Output dir to query for configs, models, etc.')

    # Listing
    parser.add_argument('--list',       '-l', action='store_true',              help="List indexed output directories")

    # Args for evaluating counterfactuals
    parser.add_argument('--eval',       '-e',       action='store_true', help="Evaluate counterfactuals")
    parser.add_argument('--all',        '-a',       action='store_true', help="Compute all metrics")
    parser.add_argument('--im1',        '-im1',     action='store_true')
    parser.add_argument('--im2',        '-m2',      action='store_true')
    parser.add_argument('--tcv',        '-tcv',     action='store_true', help="Target-class validity")
    parser.add_argument('--en',         '-en',      action='store_true', help="Elastic net sparsity measure")
    parser.add_argument('--lvs',        '-lvs',     action='store_true')
    parser.add_argument('--oracle',     '-or',      action='store_true')
    parser.add_argument('--fid',        '-fid',     action='store_true')

    parser.add_argument('--n_samples',  '-n', type=int, default=-1,             help="Limit number of samples used for computing scores")

    # Args for printing
    parser.add_argument('--print',       '-p',       action='store_true', help="Print results")

    args    = parser.parse_args()
    if args.list: 
        import listing
        listing.fn(args)

    # Load configs 
    if args.eval or args.print:
        import config
        cfgs = config.get_configs(args)
        for cfg in cfgs: config.configure_output_dir(cfg)

    # Evaluate methods 
    if args.eval:
        import evaluate
        for cfg in cfgs: 
            # Load data
            check_cfg(cfg)
            data = Dataset(cfg)

            # Evaluate counterfactuals
            evaluate.fn(args, cfg, data)
    
    # Plot results
    if args.print:
        import print_results
        for cfg in cfgs:
            print_results.fn(args, cfg)

if __name__ == "__main__":
    main()


