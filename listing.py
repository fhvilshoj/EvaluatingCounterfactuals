import os
import re
from natsort import natsorted
import configparser

def list_dir(args):
    dst         = args.output_dir
    dirs        = [os.path.join(dst, f) for f in os.listdir(dst) if os.path.isdir(os.path.join(dst, f))]
    dir_list    = natsorted(dirs)
    dir_list    = [(i, f) for i, f in enumerate(dir_list)]

    if args.query is not None: dir_list = [t for t in dir_list if re.search(args.query, t[1])]

    return dir_list

def fn(args, *_):
    print("> Listing dirs")
    files = list_dir(args)
    for t in files: print("%03i: %s" % t)
    

    

