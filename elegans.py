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
from util.points import classify_kmeans
import numpy as np
from elegans.result import Result
from elegans.funcs import _f0, _f1, groups_to_masks, load_details, load_og, load_predictions, load_saved_model, make_group_mask, make_predictions, read_dataset_images, save_details, save_og, save_predictions


def process_single_2d(item, model_pred, detail, og_2d_mask, og_source, thresh, num_points):
    '''Process a single data item.'''
    result = Result()
    
    #try:
    # Crop the base mask down. Make sure the depth has not been changed
    if int(detail['roid']) != 51:
        print("Error with no / incorrect ROI for:", item.path)
        return

    y = int(float(detail['roiy']))
    x = int(float(detail['roix']))
    w = int(float(detail['roiwh']))
    og_source = og_source[:, y:y+w, x:x+w]
    (pred, rot, trans, stretch, sigma) = model_pred
    im = torch.squeeze(pred)

    # With basic normalisation and 200 points, all values are halved so multiply by two
    # We want to reject anything below the FWHM of a single Gaussian blob
    if thresh <= 0:
        fwhm = (1.0 / (2 * math.pow(sigma, 2) * math.pi)) * math.exp(0) / 2
        result.thresh2d = (normaliser.factor / num_points) * fwhm
    else:
        result.thresh2d = thresh

    # Work out the 2D Scores
    new_2d_mask = im.numpy()
    new_2d_mask = np.where(new_2d_mask > result.thresh2d, 1, 0)
    og_2d_mask = np.where(og_2d_mask != 0, 1, 0)

    jacc = np.sum(new_2d_mask * og_2d_mask) / (np.sum(new_2d_mask) + np.sum(og_2d_mask) - np.sum(new_2d_mask * og_2d_mask))
    result.jacc2d = jacc
    
    print(item.path, "Thresh", result.thresh2d, "2D Jacc", result.jacc2d)

    #except Exception as e:
    #    print("Exception in process_single", e)
    
    return result


def process_single_3d(item, model_pred, detail, og_3d_mask, og_source, group_mask, thresh, num_points):
    result = Result()
    # Render out to 3D for mask check
    # We are using the normaliserbasic3d by default

    x = int(float(detail['roix']))
    y = int(float(detail['roiy']))
    w = int(float(detail['roiwh']))
    (pred, rot, trans, stretch, sigma) = model_pred
    og_source = og_source[:, y:y+w, x:x+w]
    og_3d_mask = og_3d_mask[:, y:y+w, x:x+w]
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


def process_2d_all(args, preds, set_test, details, og_2d_masks, og_sources, num_points):
    factor = (args.max_thresh - args.min_thresh) / args.num_samples
    results = []
    thresher = [args.min_thresh + float(i * factor) for i in range(args.num_samples)]

    for thresh in thresher:
        tresults = []
    
        for vidx in range(len(set_test)):
            detail = details[vidx]
            item = set_test.__getitem__(vidx)
            pred = preds[vidx]
            og_2d_mask = og_2d_masks[vidx]
            og_source = og_sources[vidx]
            tresults.append(process_single_2d(item, pred, detail, og_2d_mask, og_source, thresh, num_points))
        
        results.append((thresh, tresults))
    
    return results


def process_3d_all(args, preds, set_test, details, og_3d_masks, og_sources, group_masks, num_points):
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
            tresults.append(process_single_3d(item, pred, detail, og_3d_mask, og_source, group_mask, thresh, num_points))
        
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
    loader = ImageLoader(max_size=args.loader_size, image_path=args.dataset, presigma=False, sigma=args.sigma, file_filter="layered")
    set_test = DataSet(None, 0, loader, None)
    set_test.load(args.load + "/test_set.pickle")
    num_points = points.data.shape[0]
    image_size = (args.depth, args.height, args.width)

    # Get a name to save the files by
    fname = os.path.basename(args.load)

    if args.single:
        # Assume we've made all the support files
        preds = load_predictions("elegans/" + fname + "_predictions.pickle")
        og_sources, og_2d_masks, og_3d_masks = load_og("elegans/" + fname + "_sources.h5")
        details, removals = load_details("elegans/" + fname + "_details.pickle")
        vidx = args.idx  # random.randint(0, len(preds))
        og_source = og_sources[vidx]
        detail = details[vidx]
        item = set_test.__getitem__(vidx)
        pred = preds[vidx]
        print("Input Image", item.path)
        input_image = load_fits(item.path)

        x = int(float(detail['roix']))
        y = int(float(detail['roiy']))
        w = int(float(detail['roiwh']))
        cropped_source = og_source[:, y:y+w, x:x+w]
        cropped_source = np.flip(cropped_source, axis=1)

        if args.three:
            with h5py.File("elegans/" + fname + "_group_masks.h5", 'r') as hf:
                group_masks = np.array(hf['group_masks'])
                group_mask = group_masks[vidx]
                og_3d_mask = og_3d_masks[vidx]
                group_mask = np.flip(group_mask, axis=2)
                result = process_single_3d(item, pred, detail, og_3d_mask, og_source, group_mask, args.thresh3d, num_points)
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
               
        #else:
        save_fits(input_image, "elegans/input_image.fits")
        save_fits(cropped_source, "elegans/og_source.fits")
        og_2d_mask = og_2d_masks[vidx]
        save_fits(og_2d_mask, "elegans/og_2d_mask.fits")
        result = process_single_2d(item, pred, detail, og_2d_mask, og_source, args.thresh, num_points)
        (prediction, rot, trans, stretch, sigma) = pred
        save_fits(prediction, "elegans/pred_2d.fits")
        new_2d_mask = np.where(prediction > args.thresh, 1, 0)
        new_2d_mask = new_2d_mask.astype('uint8')
        save_fits(new_2d_mask, "elegans/pred_2d_mask.fits")
        print(result)
     
    else:
        removals = []  # Failures in the lookup so remove from testing
        
        if os.path.exists("elegans/" + fname + "_details.pickle") and os.path.exists("elegans/" + fname + "_sources.h5"):
            og_sources, og_2d_masks, og_3d_masks = load_og("elegans/" + fname + "_sources.h5")
            details, removals = load_details("elegans/" + fname + "_details.pickle")
        else:
            details, og_sources, og_2d_masks, og_3d_masks, removals = read_dataset_images(args.dataset, set_test, args.base, args.rep)
            save_details("elegans/" + fname + "_details.pickle", details, removals)
            save_og("elegans/" + fname + "_sources.h5", og_sources, og_2d_masks, og_3d_masks)

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

        if args.three:
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
                results = process_3d_all(args, preds, set_test, details, og_3d_masks, og_sources, group_masks, num_points)
    
            with open("elegans/" + fname + '_celegans_results_3d.pickle', 'wb') as f:
                data = pickle.dump(results, f)

        else:
            results = process_2d_all(args, preds, set_test, details, og_2d_masks, og_sources, num_points)
            
            with open("elegans/" + fname + '_celegans_results_2d.pickle', 'wb') as f:
                data = pickle.dump(results, f)
