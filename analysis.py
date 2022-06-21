
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
import argparse
import os
import torch.nn as nn
from data import loader, imageload, sets, buffer, batcher
from net.renderer import Splat
from util.image import NormaliseBasic, NormaliseNull
from util.math import TransTen, VecRotTen
from util.plyobj import load_obj
from scipy.cluster.vq import kmeans
from vedo import Points, show
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
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="The width of the input and output images \
                          (default: 128).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=128,
        help="The height of the input and output images \
                          (default: 128).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=16,
        help="The depth of the input and output images \
                          (default: 16).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="The sigma to render for prediction (default 1.0)",
    )
    parser.add_argument(
        "--max-trans",
        type=float,
        default=0.5,
        help="The maximum translation amount (default 0.5)",
    )
 
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    points = load_obj(args.final)
    points = points.to_ten()

    data_loader = imageload.ImageLoader(image_path=args.data)
    set_test = sets.DataSet(sets.SetType.TEST, 100, data_loader)
    image_size = (args.depth, args.height, args.width)
    buffer_test = buffer.BufferImage(set_test, buffer_size=1, image_size=image_size, device=device)

    print("Data Item", set_test[0].path, set_test[0].graph)
    
    #set_test.load(args.savedir + "/test_set.pickle")
    #data_loader.load(args.savedir + "/train_data.pickle")

    # Load our model and make a prediction
    model = load_model(args.savedir + "/model.tar", device)

    if os.path.isfile(args.savedir + "/checkpoint.pth.tar"):
        # TODO - For some reason, points always comes back null - probably because it doesn't save properly :/
        (model, _, _, _, _, _, prev_args) = load_checkpoint(
            model, args.savedir, "checkpoint.pth.tar", device
        )
        model.to(device)
        print("Loaded model", model)
    else:
        print("Error - need to pass in a model")


    with torch.no_grad():
        model.eval()

        # Assuming normalisation for now as we never turn it off
        normaliser_out = NormaliseBasic()
        normaliser_in = NormaliseBasic()

        # We'd like a batch rather than a similar issue.
        batcher = batcher.Batcher(buffer_test, batch_size=1)
        model.set_sigma(args.sigma)
        ddata = batcher.__next__()
        graph = ddata.graph.squeeze().numpy()
        graph = graph[:, :3]
        print("Data Graph", graph)

        # Offsets is essentially empty for the test buffer.
        target = ddata.data

        target_shaped = normaliser_in.normalise(
            target.reshape(
                1,
                1,
                args.depth,
                args.height,
                args.width,
            )
        )

        output = normaliser_out.normalise(model(target_shaped, points))
        output = output.reshape(
            1,
            1,
            args.depth,
            args.height,
            args.width,
        )

        tparams = model.get_rots()[0]

        # Now transform the points with a splat.
        ss = nn.Softsign()

        tx = ss(tparams[3]) * args.max_trans
        ty = ss(tparams[4]) * args.max_trans
        tz = ss(tparams[5]) * args.max_trans

        splat = Splat(device=device, size=image_size)

        r = VecRotTen(tparams[0], tparams[1], tparams[2])
        t = TransTen(tx, ty, tz)

        points = splat.transform_points(points, r, t)


    # Visualise the points
    points = points.data.squeeze().numpy()
    points = points[:, :3]

    groups, _ = kmeans(points, 4)
    print(groups)
    points_vedo = Points(points).c('green2')
    groups_vedo = Points(groups, r=12).c('blue2')
    graph_vedo = Points(graph, r=12).c('red2')

    show(points_vedo, groups_vedo, graph_vedo, __doc__, axes=1, viewup='y').close()