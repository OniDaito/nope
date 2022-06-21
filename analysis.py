
""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

analysis.py - visualisation of the various parts of the worm
nope stuff
"""

import torch
import numpy as np
import math
import random
import argparse
import os
import sys
from data import loader, imageload, sets
from util.plyobj import load_obj
from scipy.cluster.vq import kmeans
from vedo import np, Points, show
from util.loadsave import load_checkpoint, load_model


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="NOPE Analysis")

    parser.add_argument(
        "--final", default="./test.obj", help="The path to the final obj."
    )
    parser.add_argument(
        "--savedir", default="./run", help="The path to the trained net."
    )
    parser.add_argument(
        "--data", default="./images", help="The path to the images we used on this network."
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
 
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    data_loader = imageload.ImageLoader(image_path=args.data)
    set_test = sets.DataSet(sets.SetType.TEST, 100, data_loader)
    
    #set_test.load(args.savedir + "/test_set.pickle")
    #data_loader.load(args.savedir + "/train_data.pickle")

    model = None
    model = load_model(args.savedir + "/model.tar", device)

    if os.path.isfile(args.savedir + "/" + args.savename):
        (model, points, _, _, _, _, prev_args) = load_checkpoint(
            model, args.savedir, args.savename, device
        )
        model.to(device)
        print("Loaded model", model)
    else:
        print("Error - need to pass in a model")

    with torch.no_grad():
        model.eval()
  
    # Visualise the points
    points = load_obj(args.final)
    pnup = points.to_ten().data.squeeze().numpy()
    pnup = pnup[:, :3]

    print(pnup.shape)
    groups, _ = kmeans(pnup, 4)
    print(groups)
    pts = Points(pnup).c('green2')
    gpts = Points(groups, r=12).c('blue2')

    show(pts, gpts, __doc__, axes=1, viewup='y').close()