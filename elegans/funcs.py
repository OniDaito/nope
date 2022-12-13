"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

funcs.py - take in an image and make a prediction by producing a mask.

"""
import csv
import os
import pickle
from numba import jit
import numpy as np
import torch
import h5py
from tqdm import tqdm
from data.loader import ItemType
from net.renderer import Splat
from net.renderer3d import Splat as Splat3D
from util.image import NormaliseBasic, NormaliseNull, load_fits
from util.loadsave import load_checkpoint, load_model
from util.math import PointsTen, TransTen3D
from util.plyobj import load_obj, load_ply


def find_details(dataset, path, base, rep):
    '''
    CSV / Dict / Dataset format
    ogsource,ogmask,fitssource,fitsmask,annolog,annodat,newsource,newmask,roix,roiy,roiz,roiwh,roid,back
    Note that our dataset has the original paths. We will need to use base and rep.
    '''
    tpath = path.replace(rep, base)

    for d in dataset:
        if d['newsource'] == tpath:
            return d

    return None


@jit(nopython=True)
def _f0(mask: np.ndarray, og_source: np.ndarray, thresh: float):
    new_3d_mask = np.where(mask > thresh, 1, 0)
    new_score = np.sum(og_source * new_3d_mask)
    return (new_3d_mask, new_score)


@jit(nopython=True)
def _f1(og_source: np.ndarray, og_3d_mask: np.ndarray, new_3d_all_mask: np.ndarray):
    og3d = np.where(og_3d_mask != 0, 1, 0)
    og_3d_asi_mask = np.where(og_3d_mask == 1, 1, 0)
    og_3d_asj_mask = np.where(og_3d_mask == 2, 1, 0)
    jacc = np.sum(new_3d_all_mask * og3d) / ((np.sum(new_3d_all_mask) + np.sum(og3d) - np.sum(new_3d_all_mask * og3d)) + 0.0001)  # Avoid div by zero
    og_asi_score = np.sum(og_source * og_3d_asi_mask)
    og_asj_score = np.sum(og_source * og_3d_asj_mask)
    return (jacc, og_asi_score, og_asj_score)


def load_saved_model(args, image_height, image_width, device):
    if args.load and os.path.isfile(args.load + "/" + args.checkpoint_name):
        model = load_model(args.load + "/model.tar")
        (model, points_tensor, _, _, _, _, prev_args) = load_checkpoint(
            model, args.load, args.checkpoint_name, device
        )
        model = model.to(device)
        model.eval()
    
        # Create a splat with the correct image dimensions
        splat = Splat(size=(image_height, image_width), device=device)
        model.set_splat(splat)
        normaliser = NormaliseNull()

        if prev_args.normalise_basic:
            normaliser = NormaliseBasic()

        points = PointsTen().from_tensor(points_tensor)

        '''if args.points != "":
            points = PointsTen()
            if "ply" in args.points:
                points.from_points(load_ply(args.points))
            else:
                points.from_points(load_obj(args.points))

            points_tensor = points.data'''

        return (model, points, points_tensor, device, normaliser)


def make_predictions(model, points, test_set, device):
    ''' Perform the model predictions over the given test set'''
    model.eval()
    model_preds = []
    print("Making Predictions")

    with torch.no_grad():
        normaliser = NormaliseBasic()

        for item in tqdm(test_set):
            assert item.type == ItemType.FITSIMAGE
            input_image = load_fits(item.path)
            dimmed = torch.unsqueeze(input_image, dim=0)  # expand to 3 dims
            batch = torch.unsqueeze(dimmed, dim=0)
            normed = normaliser.normalise(batch)
            normed = normed.to(device)
            x = normaliser.normalise(model.forward(normed, points))
            (rot, trans, stretch, sigma) = model.final_params(model._final[0])
            sigma = float(sigma)
            x = torch.squeeze(x)
            model_preds.append((x, rot, trans, stretch, sigma))

    return model_preds


def save_predictions(filename, preds):
    with open(filename, 'wb') as f:
        pickle.dump(preds, f)


def load_predictions(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def make_group_mask(model_pred, points, image_size, gmasks, device):
    ''' Given a model prediction, render to 4 3D volumes we can then use
    as the mask. Each volume is one group on it's own.'''
    splat3d = Splat3D(size=image_size,  device=device)

    (_, rot, trans, stretch, sigma) = model_pred
    with torch.no_grad():
        trans3d = TransTen3D(trans.x, trans.y, torch.tensor([0], dtype=torch.float32, device=device))
        sigma = float(sigma)
        group_masks = np.zeros((4, image_size[0], image_size[1], image_size[2]))

        for gid in range(4):
            gmask = gmasks[gid]
            res3d = splat3d.render_rot_mat(points, rot, trans3d, stretch, gmask, sigma)
            res3d = torch.unsqueeze(res3d, axis=0)
            res3d = torch.unsqueeze(res3d, axis=0)

            new_3d_mask = np.flip(res3d.squeeze().squeeze().numpy(), 1)
            group_masks[gid] = new_3d_mask

    return group_masks


def groups_to_masks(groups, points):
    '''Create masks from the groups.'''
    gmasks = []

    # Render each group on it's own, perform a count.
    for gid in range(4):
        gmask = points.data.new_full([points.data.shape[0], 1, 1], fill_value=0.0)

        for ip, g in enumerate(groups):
            if g == gid:
                gmask.data[ip][0][0] = 1.0
        gmasks.append(gmask)

    return gmasks


def read_dataset_images(dataset_path, test_set, base, rep):
    '''Do all the CSV and FITS images reading, creating three 
    big numpy arrays. ''' 
    assert(os.path.exists(dataset_path + "/master_dataset.csv"))
    rvals = []
    csv_path = dataset_path + "/master_dataset.csv"
    dataset = []

    if os.path.exists(csv_path):
        with open(csv_path, mode='r') as f:
            csv_file = csv.DictReader(f)

            for line in csv_file:
                dataset.append(line)

    print("Loading FITS images and creating sources and masks.")
    # TODO - pass in dimensions
    og_sources = []
    og_2d_masks = []
    og_3d_masks = []
    removals = []

    for idx, item in enumerate(tqdm(test_set)):
        details = find_details(dataset, item.path, base, rep)

        if details is not None:
            og_sources.append(load_fits(details['fitssource'].replace(base, rep), flip=True).numpy())
            og_2d_masks.append(load_fits(details['newmask'].replace(base, rep)).numpy())
            og_3d_masks.append(load_fits(details['fitsmask'].replace(base, rep), flip=True).numpy())  # Flip before crop then flip back
            rvals.append(details)
        else:
            print("Failure on",  item.path)
            removals.append(idx)

    return (rvals, np.array(og_sources), np.array(og_2d_masks), np.array(og_3d_masks), removals)
 

def save_og(filename, og_sources, og_2d_masks, og_3d_masks):
    with h5py.File(filename, 'w') as hf:
        s = og_sources.shape[0]
        d = og_sources.shape[1]
        h = og_sources.shape[2]
        w = og_sources.shape[3]

        og_sources_hf = hf.create_dataset("og_sources", (s, d, h, w),
            maxshape=(s, d, h, w),
            chunks=(1, d, h, w))
        
        og_sources_hf[:] = og_sources

        s = og_2d_masks.shape[0]
        h = og_2d_masks.shape[1]
        w = og_2d_masks.shape[2]

        og_2d_masks_hf = hf.create_dataset("og_2d_masks", (s, h, w),
            maxshape=(s, h, w),
            chunks=(1, h, w))

        og_2d_masks_hf[:] = og_2d_masks

        s = og_3d_masks.shape[0]
        d = og_3d_masks.shape[1]
        h = og_3d_masks.shape[2]
        w = og_3d_masks.shape[3]

        og_3d_masks_hf = hf.create_dataset("og_3d_masks", (s, d, h, w),
            maxshape=(s, d, h, w),
            chunks=(1, d, h, w))

        og_3d_masks_hf[:] = og_3d_masks


def load_og(filename):
    with h5py.File(filename, 'r') as hf:
        og_sources_hf = np.array(hf['og_sources'])
        og_2d_masks_hf = np.array(hf['og_2d_masks'])
        og_3d_masks_hf = np.array(hf['og_3d_masks'])

    return (og_sources_hf, og_2d_masks_hf, og_3d_masks_hf)


def save_details(filename, details, removals):
    with open(filename, 'wb') as f:
        pickle.dump((details, removals), f)


def load_details(filename):
    with open(filename, 'rb') as f:
        details, removals = pickle.load(f)
    return details, removals