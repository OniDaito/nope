"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

elegans.py - take in an image and make a prediction by producing a mask.

Images must already be processed and ready to be converted. That includes
any ROI from the original images or conversions.

Example usage:

python elegans.py --input-image ./test/images/celegans/layered.fits \
--source-volume ./test/images/celegans/source.fits \
--load /media/proto_working/runs/2022_10_31

python elegans.py --load /media/proto_working/runs/2022_08_18_2 \
    --dataset /media/proto_backup/wormz/queelim/dataset_2d_basic \
    --loader-size 2750 \
    --base /phd/wormz/ \
    --rep /media/proto_backup/wormz/ \
    --no-cuda

"""

import argparse
import math
import os
import pickle
import torch
import h5py
import random
from tqdm import tqdm
from data.imageload import ImageLoader
from data.sets import DataSet
from util.image import load_fits, save_fits
from util.math import PointsTen
from util.plyobj import load_obj, load_ply
from util.points import classify_kmeans
import numpy as np
from elegans.result import Result
from elegans.funcs import _f0, _f1, groups_to_masks, load_details, load_og, load_predictions, load_saved_model, make_group_mask, make_predictions, read_dataset_images, save_details, save_og, save_predictions


def resize_mask(img : torch.Tensor, size : tuple):
    new_img = np.zeros(size, dtype=np.float32)
    og_size = img.shape
    zd = og_size[0] / size[0]
    yd = og_size[1] / size[1]
    xd = og_size[2] / size[2]

    for z in range(size[0]):
        sz = int(z * zd)

        for y in range(size[1]):
            sy = int(y * yd)

            for x in range(size[2]):
                sx = int(x * xd)
                new_img[z][y][x] = img[sz][sy][sx]

    return new_img


def process_single_3d(item, model_pred, detail, og_3d_mask, og_source, group_mask, thresh, num_points, acrop, image_size):
    result = Result()
    # Render out to 3D for mask check
    # We are using the normaliserbasic3d by default

    w = int(float(detail['roiwh']))
    a = acrop
    d = int((w - a)/ 2)

    x = int(float(detail['roix'])) + d
    y = int(float(detail['roiy'])) + d

    (pred, rot, trans, stretch, sigma) = model_pred
    og_source = og_source[:, y:y+a, x:x+a]
    og_3d_mask = og_3d_mask[:, y:y+a, x:x+a]
    og_3d_mask = np.flip(og_3d_mask, axis=1)  # Flip after cropping


    # Now work on the 3D scores
    if thresh <= 0:
        fwhm = (1.0 / (2 * math.pow(sigma, 3) * math.pi)) * math.exp(0) / 2
        result.thresh3d = (100.0 / num_points) * fwhm
    else:
        result.thresh3d = thresh  # NOTE - for 3D the lowest thresh is 10 times lower roughly (3D spread)

    new_3d_all_mask = np.zeros(og_3d_mask.shape)
    
    for gid in range(4):
        new_3d_mask = np.array(group_mask[gid])
        # Perform our normalisation here - based on NormaliseBasic3D but taking into account the
        # differing numbers of points. x / total_intensity * factor
        factor = 100.0 / num_points
        new_3d_mask *= factor
        
        # Resize if needed 
        if new_3d_mask.shape[0] != og_source.shape[0] or new_3d_mask.shape[1] != og_source.shape[1] \
            or new_3d_mask.shape[2] != og_source.shape[2]:
            new_3d_mask = resize_mask(new_3d_mask, og_source.shape)

        new_3d_mask, new_score = _f0(new_3d_mask, og_source, result.thresh3d)
        new_3d_all_mask += new_3d_mask
        result.new_scores.append(new_score)
    
    new_3d_all_mask = np.where(new_3d_all_mask != 0, 1, 0)
    
    # Now get the scores for ASI and ASJ
    jacc, og_asi_score, og_asj_score = _f1(og_source, og_3d_mask, new_3d_all_mask)
    result.jacc3d = jacc
    result.og_asi_score = og_asi_score
    result.og_asj_score = og_asj_score

    print(item.path, "3D =OG Scores ASI / ASJ:", result.og_asi_score, ",", result.og_asj_score,  "New Scores:", result.new_scores, "Jacc:", jacc, "Thresh:", result.thresh3d)

    return result


def process_3d_all(args, preds, set_test, details, og_3d_masks, og_sources, group_masks, num_points, image_size):
    factor = (args.max_thresh - args.min_thresh) / float(args.num_samples)
    results = []
    thresher = [args.min_thresh + float(i) * factor for i in range(args.num_samples)]

    for thresh in thresher:
        tresults = []
        print("thresh", thresh)
    
        for vidx in range(len(set_test)):
            detail = details[vidx]
            item = set_test.__getitem__(vidx)
            pred = preds[vidx]
            og_3d_mask = og_3d_masks[vidx]
            og_source = og_sources[vidx]
            group_mask = group_masks[vidx]
            group_mask = np.flip(group_mask, axis=2)
            tresults.append(process_single_3d(item, pred, detail, og_3d_mask, og_source, group_mask, thresh, num_points, args.acrop, image_size))
        
        results.append((thresh, tresults))

    return results


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="HOLLy C. Elegans prog.")

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--single", action="store_true", default=False, help="Do a single test and save the fits"
    )
    parser.add_argument(
        "--three", action="store_true", default=False, help="Do 3D masks and generate counts"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed (default: 1)"
    )
    # TODO - remove this when we fix the wiggle CSV crop stuff
    parser.add_argument(
        "--acrop", type=int, default=200, help="Actual final crop size of the dataset"
    )
    parser.add_argument(
        "--idx", type=int, default=1, help="If single, which one?"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.8, help="Sigma to use"
    )
    parser.add_argument(
        "--thresh", type=float, default=-1, help="Override the thresh",
    )
    parser.add_argument(
        "--thresh3d", type=float, default=-1, help="Override the thresh for 3D",
    )
    parser.add_argument(
        "--min-thresh", type=float, default=0.00001, help="Minimum Threshold",
    )
    parser.add_argument(
        "--max-thresh", type=float, default=0.001, help="Maximum Threshold",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of samples to generate."
    )
    parser.add_argument(
        "--width", type=int, default=200, help="Image width"
    )
    parser.add_argument(
        "--height", type=int, default=200, help="Image height"
    )
    parser.add_argument(
        "--depth", type=int, default=51, help="Image depth"
    )
    parser.add_argument(
        "--loader-size", type=int, default=1000, help="How big was the data loader?"
    )
    parser.add_argument(
        "--test-size", type=int, default=100, help="How big was the test set?"
    )
    parser.add_argument(
        "--load", default="./save", help="The name for checkpoint save directory."
    )
    parser.add_argument(
        "--dataset", default="", help="The path to the dataset for this particular run"
    )
    parser.add_argument(
        "--points", default="", help="Points to use (default: none)."
    )
    parser.add_argument('--base', default="")
    parser.add_argument('--rep', default="")
    
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint.pth.tar",
        help="The name for checkpoint save file.",
    )

    # Initial setup of PyTorch
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Load all the base data
    model, points, points_tensor, device, normaliser = load_saved_model(args, args.depth, args.height, args.width, device)
    
    if os.path.exists(args.points):
        if "obj" in args.points:
            points = load_obj(objpath=args.points)
        elif "ply" in args.points:
            points = load_ply(args.points)
        
        points = PointsTen(device=device).from_points(points)

    loader = ImageLoader(size=args.loader_size, image_path=args.dataset, presigma=False, sigma=args.sigma)
    set_test = DataSet(None, 0, loader, None)
    set_test.load(args.load + "/test_set.pickle")
    num_points = points.data.shape[0]
    image_size = (args.depth, args.height, args.width)

    # Get a name to save the files by
    fname = os.path.basename(args.load)

    if args.single:
        # Assume we've made all the support files
        preds = load_predictions("elegans/" + fname + "_predictions.pickle")
        og_sources, og_3d_masks = load_og("elegans/" + fname + "_sources.h5")
        details, removals = load_details("elegans/" + fname + "_details.pickle")
        vidx = args.idx  # random.randint(0, len(preds))
        og_source = og_sources[vidx]
        detail = details[vidx]
        item = set_test.__getitem__(vidx)
        pred = preds[vidx]
        print("Input Image", item.path)
        input_image = load_fits(item.path)

        w = int(float(detail['roiwh']))
        a = args.acrop
        d = int((w - a)/ 2)

        x = int(float(detail['roix'])) + d
        y = int(float(detail['roiy'])) + d

        cropped_source = og_source[:, y:y+a, x:x+a]
        cropped_source = np.flip(cropped_source, axis=1)

        with h5py.File("elegans/" + fname + "_group_masks.h5", 'r') as hf:
            group_masks = np.array(hf['group_masks'])
            group_mask = group_masks[vidx]
            og_3d_mask = og_3d_masks[vidx]
            group_mask = np.flip(group_mask, axis=2)
            result = process_single_3d(item, pred, detail, og_3d_mask, og_source, group_mask, args.thresh3d, num_points, a, image_size)
            (prediction, rot, trans, stretch, sigma) = pred
            og_3d_mask = og_3d_mask[:, y:y+w, x:x+w]
            og_3d_mask = np.flip(og_3d_mask, axis=1)
            fmask = np.sum(group_mask, axis=0)
            fmask = np.flip(fmask, axis=0)
            nmask, score = _f0(fmask, cropped_source, args.thresh3d)
            nmask = nmask.astype('uint8')

            save_fits(fmask, "elegans/pred_3d.fits")
            save_fits(og_3d_mask, "elegans/og_3d_mask.fits")
            #save_fits(prediction, "elegans/prediction.fits")
            save_fits(nmask, "elegans/pred_3d_mask.fits")
     
    else:
        removals = []  # Failures in the lookup so remove from testing
        
        if os.path.exists("elegans/" + fname + "_details.pickle") and os.path.exists("elegans/" + fname + "_sources.h5"):
            og_sources, og_3d_masks = load_og("elegans/" + fname + "_sources.h5")
            details, removals = load_details("elegans/" + fname + "_details.pickle")
        else:
            details, og_sources, og_3d_masks, removals = read_dataset_images(args.dataset, set_test, args.base, args.rep)
            save_details("elegans/" + fname + "_details.pickle", details, removals)
            save_og("elegans/" + fname + "_sources.h5", og_sources, og_3d_masks)

        # Get the predictions
        preds = None
        print("Removals", removals)

        for i in sorted(removals, reverse=True):
            set_test.remove(i)

        if os.path.exists("elegans/" + fname + "_predictions.pickle"):
            preds = load_predictions("elegans/" + fname + "_predictions.pickle")
       
        else:
            preds = make_predictions(model, points, set_test, device)
            save_predictions("elegans/" + fname + "_predictions.pickle", preds)

        base_points = points.get_points()
        groups = classify_kmeans(base_points)
        gmasks = groups_to_masks(groups, points)
        
        if not os.path.exists("elegans/" + fname + "_group_masks.h5"):
            # Render the 3D masks, saving each one to the HDF5 file to save memory
            with h5py.File("elegans/" + fname + "_group_masks.h5", 'w') as hf:
                s = len(set_test)
                d = image_size[0]
                h = image_size[1]
                w = image_size[2]

                print("Creating Group Masks...")
                group_masks = hf.create_dataset("group_masks", (s, 4, d, h, w),
                    maxshape=(s, 4, d, h, w),
                    chunks=(1, 4, d, h, w))
        
                for pidx, pred in enumerate(tqdm(preds)):
                    group_masks[pidx] = make_group_mask(pred, points, image_size, gmasks, device)

        with h5py.File("elegans/" + fname + "_group_masks.h5", 'r') as hf:
            group_masks = np.array(hf['group_masks'])
            results = process_3d_all(args, preds, set_test, details, og_3d_masks, og_sources, group_masks, num_points, image_size)

        with open("elegans/" + fname + '_celegans_results_3d.pickle', 'wb') as f:
            data = pickle.dump(results, f)
