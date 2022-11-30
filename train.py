""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

train.py - an attempt to find the 3D shape from an image.
To train a network, use:
  python train.py <OPTIONS>

See the README file and the __main__ function for the
various options.

"""

from email.mime import base
from multiprocessing import reduction
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import argparse
import os
import sys
from util.points import init_points
from util.loadsave import save_checkpoint, save_model
from data.loader import Loader
from data.imageload import ImageLoader
from data.sets import DataSet, SetType
from data.buffer import Buffer, BufferImage
from data.batcher import Batcher
from stats import stats as S
from net.renderer import Splat
from net.net import Net
from net.model import Model
from util.image import NormaliseNull, NormaliseBasic
from globals import DTYPE, badness
from torch import autograd

def calculate_loss(target: torch.Tensor, output: torch.Tensor):
    """
    Our loss function, used in train and test functions.

    Parameters
    ----------

    target : torch.Tensor
        The target, properly shaped.

    output : torch.Tensor
        The tensor predicted by the network, not shaped

    Returns
    -------
    Loss
        A loss object
    """

    loss = F.l1_loss(output, target, reduction="sum")
    
    # Loss can hit this if things have moved too far, so redo loss
    if badness(loss):
        assert(False, "Badness in Loss")

    return loss


def test(
    args,
    model,
    buffer_test: Buffer,
    epoch: int,
    step: int,
    points_model: Model,
    sigma: float,
    write_fits: bool
):
    """
    Switch to test / eval mode and do some recording to our stats
    program and see how we go.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    model : torch.nn.Module
        The main net model
    buffer_test : Buffer
        The buffer that represents our test data.
    epoch : int
        The current epoch.
    step : int
        The current step.
    points : PointsTen
        The current PointsTen being trained.
    sigma : float
        The current sigma.
    write_fits : bool
        Write the intermediate fits files for analysis.
        Takes up a lot more space. Default - False.
    Returns
    -------
    None
    """

    # Put model in eval mode
    model.eval()

    # Which normalisation are we using?
    normaliser_out = NormaliseNull()
    normaliser_in = NormaliseNull()

    if args.normalise_basic:
        normaliser_out = NormaliseBasic()
        normaliser_in = NormaliseBasic()  # NormaliseWorm()

    image_choice = random.randrange(0, args.batch_size)
    # We'd like a batch rather than a similar issue.
    batcher = Batcher(buffer_test, batch_size=args.batch_size)
    rots_in = []  # Save rots in for stats
    trans_in = []
    rots_out = []  # Collect all rotations out
    stretch_in = []
    model.set_sigma(sigma)

    if args.objpath != "":
        # Assume we are simulating so we have rots to save
        S.watch(rots_in, "rotations_in_test")
        S.watch(rots_out, "rotations_out_test")
        S.watch(trans_in, "translation_in_test")
        S.watch(stretch_in, "stretch_in_test")


    for batch_idx, ddata in enumerate(batcher):
        # turn off grads because for some reason, memory goes BOOM!
        with torch.no_grad():
            # Offsets is essentially empty for the test buffer.
            target = ddata.data
            target_shaped = normaliser_in.normalise(
                target.reshape(
                    args.batch_size,
                    1,
                    args.image_depth,
                    args.image_height,
                    args.image_width,
                ) #, sigma
            )

            output = normaliser_out.normalise(model(target_shaped, points_model.data))
            output = output.reshape(
                args.batch_size,
                1,
                args.image_depth,
                args.image_height,
                args.image_width,
            )

            '''if not (torch.all(torch.isnan(target) == False)):
                print("target nans", target)
                assert(False)

            if not (torch.all(torch.isnan(model._final) == False)):
                print("final nans", model._final)
                assert(False)
            
            if not (torch.all(torch.isnan(output) == False)):
                print("output nans", output)
                assert(False)

            if (output.float().sum().data[0] == 0):
                print("output is all zero", output)
                assert(False)'''

            rots_out.append(model.get_rots())
            loss = calculate_loss(target_shaped, output)

            # Just save one image for now - first in the batch
            if batch_idx == image_choice:
                target = torch.squeeze(target_shaped[0])
                output = torch.squeeze(output[0])
                S.save_jpg(
                    torch.sum(target, dim=0),
                    args.savedir,
                    "in_e",
                    epoch,
                    step,
                    batch_idx,
                )
                S.save_jpg(
                    torch.sum(output, dim=0),
                    args.savedir,
                    "out_e",
                    epoch,
                    step,
                    batch_idx,
                )

                if write_fits:
                    S.save_fits(target,  args.savedir, "in_e", epoch, step, batch_idx)
                    S.save_fits(output,  args.savedir, "out_e", epoch, step, batch_idx)
          
                ps = model._final.shape[1] - 1
                sp = nn.Softplus(threshold=12)
                sig_out = torch.tensor(
                    [torch.clamp(sp(x[ps]), max=14) for x in model._final]
                )
                S.watch(sig_out, "sigma_out_test")

            # soft_plus = torch.nn.Softplus()
            S.watch(loss, "loss_test")  # loss saved for the last batch only.
            if args.objpath != "":
                # Assume we are simulating so we have rots to save
                rots_in.append(ddata.rotations)
                trans_in.append(ddata.translations)
                stretch_in.append(ddata.stretches)

    buffer_test.set.shuffle()
    model.train()
    return loss


def cont_sigma(args, current_epoch: int, sigma: float, sigma_lookup: list) -> float:
    """
    If we are using _cont_sigma, we need to work out the linear
    relationship between the points. We call this each step.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    current_epoch : int
        The current epoch.
    sigma : float
        The current sigma.
    sigma_lookup : list
        The sigma lookup list of floats.

    Returns
    -------
    float
        The sigma to use
    """

    eps = args.epochs / (len(sigma_lookup) - 1)
    a = 0
    if current_epoch > 0:
        a = int(current_epoch / eps)
    b = a + 1
    assert b < len(sigma_lookup)
    assert a >= 0

    ssig = sigma_lookup[a]
    esig = sigma_lookup[b]

    ssize = args.train_size
    # TODO - we really should use the loader size here
    if args.aug:
        ssize = args.train_size * args.num_aug
    steps_per_epoch = ssize / args.batch_size
    steps = steps_per_epoch * eps
    cont_factor = math.pow(float(esig) / (float(ssig) + 1e-5), 1.0 / (float(steps) + 1e-5)) 
    new_sigma = sigma * cont_factor
    return new_sigma


def train(
    args,
    device,
    sigma_lookup,
    model,
    points_model,
    buffer_train,
    buffer_test,
    data_loader,
    optimiser,
):
    """
    Now we've had some setup, lets do the actual training.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    device : str
        The device to run the model on (cuda / cpu)
    sigma_lookup : list
        The list of float values for the sigma value.
    model : nn.Module
        Our network we want to train.
    points_model : Model
        The model made up of our points
    buffer_train :  Buffer
        The buffer in front of our training data.
    data_loader : Loader
        A data loader (image or simulated).
    optimiser : torch.optim.Optimizer
        The optimiser we want to use.

    Returns
    -------
    None
    """

    model.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min')

    # Which normalisation are we using?
    normaliser_out = NormaliseNull()
    normaliser_in = NormaliseNull()

    if args.normalise_basic:
        normaliser_out = NormaliseBasic()
        normaliser_in = NormaliseBasic()  # NormaliseWorm()

    sigma = sigma_lookup[0]
    test_loss = 0

    # We'd like a batch rather than a similar issue.
    batcher = Batcher(buffer_train, batch_size=args.batch_size)

    # Begin the epochs and training
    for epoch in range(args.epochs):
        data_loader.set_sigma(sigma)

        # Now begin proper
        print("Starting Epoch", epoch)
        for batch_idx, ddata in enumerate(batcher):
            #with autograd.detect_anomaly():
            target = ddata.data
            optimiser.zero_grad()

            # Shape and normalise the input batch
            target_shaped = normaliser_in.normalise(
                target.reshape(
                    args.batch_size,
                    1,
                    args.image_depth,
                    args.image_height,
                    args.image_width,
                ) #, sigma
            )
        
            with torch.autocast("cuda"): # TODO - do we need cpu device check option?
                output = normaliser_out.normalise(model(target_shaped, points_model.data))
                loss = calculate_loss(target_shaped, output)

            loss.backward()
        
            lossy = loss.item()
            optimiser.step()
    
            if badness(points_model.data.data):
                assert(False)

            # If we are using continuous sigma, lets update it here
            sigma = cont_sigma(args, epoch, sigma, sigma_lookup)
            data_loader.set_sigma(sigma)
            model.set_sigma(sigma)

            # We save here because we want our first step to be untrained
            # network
            if batch_idx % args.log_interval == 0:
                # Add watches here
                S.watch(lossy, "loss_train")
                # Temporary ignore of images in the DB
                # S.watch(target[0], "target")
                # S.watch(output[0], "output")
                S.watch(sigma, "sigma_in")

                print(
                    "Train Epoch: \
                    {} [{}/{} ({:.0f}%)]\tLoss Main: {:.6f}".format(
                        epoch,
                        batch_idx * args.batch_size,
                        buffer_train.set.size,
                        100.0 * batch_idx * args.batch_size / buffer_train.set.size,
                        lossy,
                    )
                )

                if args.save_stats:
                    test_loss = test(args, model, buffer_test, epoch, batch_idx, points_model, sigma, args.write_fits)
                    S.save_points(points_model, args.savedir, epoch, batch_idx)
                    S.update(
                        epoch, buffer_train.set.size, args.batch_size, batch_idx
                    )

                if batch_idx % args.save_interval == 0:
                    print("saving checkpoint", batch_idx, epoch)
                    save_model(model, args.savedir + "/model.tar")

                    save_checkpoint(
                        model,
                        points_model,
                        optimiser,
                        epoch,
                        batch_idx,
                        loss,
                        sigma,
                        args,
                        args.savedir,
                        args.savename,
                    )

        buffer_train.set.shuffle()
        scheduler.step(test_loss)

        for i, group in enumerate(optimiser.param_groups):
            S.watch(group['lr'], "lr" + str(i))

    # Save a final points file once training is complete
    S.save_points(points_model, args.savedir, epoch, batch_idx)
    print("saving checkpoint", batch_idx, epoch)
    save_model(model, args.savedir + "/model.tar")

    save_checkpoint(
        model,
        points_model,
        optimiser,
        epoch,
        batch_idx,
        loss,
        sigma,
        args,
        args.savedir,
        args.savename,
    )

    return points_model


def init(args, device):
    """
    Initialise all of our models, optimizers and other useful
    things before passing on to train.

    Parameters
    ----------
    args : dict
        The arguments object created in the __main__ function.
    device : str
        The device to run the model on (cuda / cpu)

    Returns
    -------
    None
    """

    # Continue training or start anew
    # Declare the variables we absolutely need
    model = None
    buffer_train = None
    buffer_test = None
    data_loader = None
    optimiser = None

    train_set_size = args.train_size
    # valid_set_size = args.valid_size
    test_set_size = args.test_size

    if args.aug:
        train_set_size = args.train_size * args.num_aug
        # valid_set_size = args.valid_size * args.num_aug
        test_set_size = args.test_size * args.num_aug

    # Sigma checks. Do we use a file, do we go continuous etc?
    # Check for sigma blur file
    sigma_lookup = [10.0, 1.25]

    if len(args.sigma_file) > 0:
        if os.path.isfile(args.sigma_file):
            with open(args.sigma_file, "r") as f:
                ss = f.read()
                sigma_lookup = []
                tokens = ss.replace("\n", "").split(",")
                for token in tokens:
                    sigma_lookup.append(float(token))

    # Setup our splatting pipeline. We use two splats with the same
    # values because one never changes its points / mask so it sits on
    # the gpu whereas the dataloader splat reads in differing numbers of
    # points.
    image_size = (args.image_depth, args.image_height, args.image_width)
    splat_in = Splat(device=device, size=image_size)
    splat_out = Splat(device=device, size=image_size)

    # Setup the dataloader - either generated from OBJ or fits
    if args.fitspath != "":
        data_loader = ImageLoader(
            size=args.train_size + args.test_size,
            image_path=args.fitspath
        )

        set_train = DataSet(
            SetType.TRAIN, train_set_size, data_loader, alloc_csv=args.allocfile
        )
        set_test = DataSet(SetType.TEST, test_set_size, data_loader)

        buffer_train = BufferImage(
            set_train,
            buffer_size=args.buffer_size,
            device=device,
            blur=True,
            image_size=image_size,
        )
        buffer_test = BufferImage(
            set_test, buffer_size=test_set_size, blur=True, image_size=image_size, device=device
        )

    elif args.objpath != "":
        data_loader = Loader(
            size=args.train_size + args.test_size,
            objpaths=[(args.objpath, 1)],
            wobble=args.wobble,
            dropout=args.dropout,
            spawn=args.spawn_rate,
            max_spawn=args.max_spawn,
            stretch=args.stretch,
            max_stretch=args.max_stretch,
            sigma=sigma_lookup[0],
            max_trans=args.max_trans,
            augment=args.aug,
            num_augment=args.num_aug,
        )

        fsize = min(data_loader.size - test_set_size, train_set_size)
        set_train = DataSet(SetType.TRAIN, fsize, data_loader, alloc_csv=args.allocfile)
        set_test = DataSet(SetType.TEST, test_set_size, data_loader)

        buffer_train = Buffer(
            set_train, splat_in, buffer_size=args.buffer_size, device=device
        )

        buffer_test = Buffer(
            set_test, splat_in, buffer_size=test_set_size, device=device
        )
    else:
        raise ValueError("Yomodelu must provide either fitspath or objpath argument.")

    # Create a model of points. We create from multiple obj/ply files and keep
    # track of the indices.
    points_model = Model()

    if len(args.startobjs) > 0:
        points_model.load_models(args.startobjs)
    else:
        tpoints = init_points(
            args.num_points, device=device, deterministic=args.deterministic, spread=0.3
        )
        points_model.add_points(tpoints, 0)

    points_model.make_ten(device=device)
    points_model.data.data.requires_grad_(requires_grad=True)

    # Create our main network
    model = Net(
        splat_out,
        max_trans=args.max_trans,
        stretch=args.stretch,
        max_stretch=args.max_stretch,
        predict_sigma=args.predict_sigma
    ).to(device)

    # Save the training data to disk so we can interrogate it later
    set_test.save(args.savedir + "/test_set.pickle")
    set_train.save(args.savedir + "/train_set.pickle")
    data_loader.save(args.savedir + "/train_data.pickle")

    variables = []
    variables.append({"params": model.parameters(), "lr": args.lr})
    
    if not args.poseonly:
        variables.append({"params": points_model.data.data, "lr": args.plr})
    
    optimiser = optim.AdamW(variables, eps=1e-04) # eps set here for float16 ness
    print("Starting new model")

    # Now start the training proper
    train(
        args,
        device,
        sigma_lookup,
        model,
        points_model,
        buffer_train,
        buffer_test,
        data_loader,
        optimiser
    )

    # TODO - out for now
    points_model.save_ply(args.savedir + "/last_points.ply")
    save_model(model, args.savedir + "/model.tar")


def check_args(args) -> bool:
    """
    Our loss function, used in train and test functions.

    Parameters
    ----------

    args : dictionary? Namespace?
        The args passed into the program, parsed by argparse

    Returns
    -------
    bool
        Is args valid?
    """

    return True


if __name__ == "__main__":
    # Training settings
    # TODO - potentially too many options now so go with a conf file?
    parser = argparse.ArgumentParser(description="PyTorch Shaper Train")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="input batch size for training \
                          (default: 20)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0004,
        help="learning rate (default: 0.0004)",
    )
    parser.add_argument(
        "--spawn-rate",
        type=float,
        default=1.0,
        help="Probabilty of spawning a point \
                          (default: 1.0).",
    )
    parser.add_argument(
        "--max-trans",
        type=float,
        default=1.0,
        help="The scalar on the translation we generate and predict \
                          (default: 1.0).",
    )
    parser.add_argument(
        "--max-spawn",
        type=int,
        default=1,
        help="How many flurophores are spawned total. \
                          (default: 1).",
    )
    parser.add_argument(
        "--save-stats",
        action="store_true",
        default=False,
        help="Save the stats of the training for later \
                          graphing.",
    )
    parser.add_argument(
        "--predict-sigma",
        action="store_true",
        default=False,
        help="Save the data used in the training (default: False).",
    )
    parser.add_argument(
        "--write-fits",
        action="store_true",
        default=False,
        help="Save fits files when saving stats (default: False).",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training."
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Run deterministically",
    )
    parser.add_argument(
        "--normalise-basic",
        action="store_true",
        default=False,
        help="Normalise with torch basic intensity divide",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training \
                          status",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=200,
        help="how many points to optimise (default 200)",
    )
    parser.add_argument(
        "--aug",
        default=False,
        action="store_true",
        help="Do we augment the data with XY rotation (default False)?",
        required=False,
    )
    parser.add_argument(
        "--poseonly",
        default=False,
        action="store_true",
        help="Do we just attempt to fit the pose?",
        required=False,
    )
    parser.add_argument(
        "--max-stretch",
        type=float,
        default=1.0,
        help="The scalar on the stretch we generate and predict \
                          (default: 1.0).",
    )
    parser.add_argument(
        "--stretch",
        default=False,
        action="store_true",
        help="Do we stretch the input data and model for it (default: False)?",
        required=False,
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=10,
        help="how many augmentations to perform per datum (default 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1000,
        help="how many batches to wait before saving.",
    )
    parser.add_argument(
        "--load",
        help="A checkpoint file to load in order to continue \
                          training",
    )
    parser.add_argument(
        "--savename",
        default="checkpoint.pth.tar",
        help="The name for checkpoint save file.",
    )
    parser.add_argument(
        "--savedir", default="./save", help="The name for checkpoint save directory."
    )
    parser.add_argument(
        "--allocfile", default=None, help="An optional data order allocation file."
    )
    parser.add_argument(
        "--sigma-file", default="", help="Optional file for the sigma blur dropoff."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="When coupled with objpath, what is the chance of \
                          a point being dropped? (default 0.0)",
    )
    parser.add_argument(
        "--plr",
        type=float,
        default=0.001,
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--wobble",
        type=float,
        default=0.0,
        help="Distance to wobble our fluorophores \
                          (default 0.0)",
    )
    parser.add_argument(
        "--fitspath",
        default="",
        help="Path to a directory of FITS files.",
        required=False,
    )
    parser.add_argument(
        "--objpath",
        default="",
        help="Path to the obj for generating data",
        required=False,
    )
    parser.add_argument(
        "--startobjs",
        default="",
        metavar="S",
        type=str,
        nargs="+",
        help="Path to the objs that forms the starting points.",
        required=False,
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=50000,
        help="The size of the training set (default: 50000)",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=320,
        help="The width of the input and output images \
                          (default: 320).",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=150,
        help="The height of the input and output images \
                          (default: 150).",
    )
    parser.add_argument(
        "--image-depth",
        type=int,
        default=25,
        help="The depth of the input and output images \
                          (default: 25).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=200,
        help="The size of the training set (default: 200)",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=200,
        help="The size of the training set (default: 200)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=40000,
        help="How big is the buffer in images? \
                          (default: 40000)",
    )
    args = parser.parse_args()

    # Stats turn on
    if args.save_stats:
        S.on(args.savedir)

    if not check_args(args):
        sys.exit(0)

    # Initial setup of PyTorch
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    print("Using device", device)

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True

    init(args, device)
    print("Finished Training")
    S.close()
