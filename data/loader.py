""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

loader.py -The Dataloader is responsible for generating data
for the DataSet and DataBuffer classes. It either generates on demand
or reads images from the disk. It isn't used directly, rather it works
as follows:

DataBuffer -> DataSet  |
DataBuffer -> DataSet  | -> DataLoader / CepLoader
DataBuffer -> DataSet  |

DataLoader provides all the data for as many DataSets and their associated
buffers as one would want. It performs no further processing such as conversion
to Tensor or normalisation. These take place at the DataSet level.

"""

import os
import random
import pickle
import array
import math
from tqdm import tqdm
from enum import Enum
from util.math import Points, Point, Mask, Trans, VecRot, Stretch
from data.item import LoaderItem, ItemSimulated, ItemType



def sort_models(tup):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort using second element of
    # sublist lambda has been used
    tup.sort(key=lambda x: x[1])
    return tup


class Loader(object):
    """Our Loader for simulated data. Given an obj file and some parameters,
    it will generate a set of points and transforms, ready for the Set."""

    def __init__(
        self,
        size=1000,
        objpaths=[("asil.obj", 1), ("asir.obj", 2), ("asjl.obj", 3), ("asjr.obj", 4)],
        dropout=0.0,
        wobble=0.0,
        spawn=1.0,
        max_spawn=1,
        sigma=1.25,
        max_trans=0.1,
        stretch=True,
        max_stretch=1.5,
        augment=False,
        num_augment=10
    ):
        """
        Create our Loader.

        Parameters
        ----------
        size : int
           The number of simulated items to generate
        dropout : float
            The chance that a point will be masked out, normalised, with
            1.0 being a certainty. Default - 0.
        wobble : float
            How far do we randomly peturb each point? Default - 0.0.
        spawn : float
            The chance that point will be rendered at this base point.
            Normalised, with 1.0 being a certainty. Default - 1.0.
        max_spawn : int
            What is the maximum number of points to be spawned at a
            single ground truth point? Default - 1.
        sigma : float
            The sigma we should set on all the data points. Default - 1.25.
        translate : bool
            Should we translate the data by a random amount?
            Default - True.
        rotate : bool
            Should we rotate the data by a random amount?
            Default - True.
        augment : bool
            Do we want to augment the data by performing rotations in the X,Y plane
            Default - False.
        num_augment : int
            How many augmentations per data-point should we use.
            Default - 10

        Returns
        -------
        self
        """

        # Total size of the data available and what we have allocated
        self.size = size
        self.counter = 0
        self.available = array.array("L")

        # Our ground truth object points
        self.gt_points = Points()
        self.class_indices = [0, 0, 0, 0, 0]

        # How far do we translate?
        self.max_trans = max_trans

        # How far do we stretch?
        self.stretch = stretch
        self.max_stretch = max_stretch

        # The rotations and translations we shall use
        self.transform_vars = array.array("d")

        # dropout masks (per sigma)
        self.masks = array.array("d")

        # What sigma of data are we at?
        self.sigma = sigma

        # Actual points we are using (generated from groundtruth)
        self.points = array.array("d")

        # How is the data chunked? (i.e how many points and masks)
        self.points_chunk = 0
        self.masks_chunk = 0

        # Augmentation - essentially a number of 2D affine rotations in XY
        self.augment = augment
        self.num_augment = num_augment

        # Paramaters for generating points
        self.dropout = dropout
        self.wobble = wobble
        self.spawn = spawn
        self._max_spawn = max_spawn  # Potentially, how many more flurophores

        # Load each model into our points 
        from util.plyobj import load_obj, load_ply
        sort_models(objpaths)

        for (modelpath, classidx) in objpaths:
            assert(classidx <= len(self.class_indices))

            if "obj" in modelpath:
                self.gt_points.cat(load_obj(objpath=modelpath))

            elif "ply" in modelpath:
                self.gt_points.cat(load_ply(modelpath))

            self.class_indices[classidx] = self.gt_points.size

        self._create_basic()

        # Set here as once we've augmented we need a new size
        if self.augment:
            self.size = size * num_augment

    def reset(self):
        """
        Reset the loader. Delete all the data and rebuild.

        Parameters
        ----------
        None

        Returns
        -------
        self
        """

        self.transform_vars = array.array("d")
        self.masks = array.array("d")
        self.points = array.array("d")
        # TODO - should somehow invalidate the sets above?
        self.available = array.array("L")
        self._create_basic()
        return self

    def remaining(self) -> int:
        """
        Return the number of items remaining that can be claimed by the
        dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
        """
        return len(self.available)

    def __next__(self) -> LoaderItem:
        if self.counter >= self.size:
            print("Reached the end of the dataloader.")
            self.counter = 0
            raise StopIteration
        else:
            rval = self.__getitem__(self.counter)
            self.counter += 1
            return rval

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return self

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, idx) -> LoaderItem:
        """
        Return the LoaderItem at the position denoted with idx.

        Parameters
        ----------
        idx : int
           The index of the LoaderItem we want.

        Returns
        -------
        LoaderItem
            The item at idx
        """
        points = Points()
        for i in range(int(self.points_chunk / 4)):
            ts = idx * self.points_chunk + i * 4
            point = Point(
                self.points[ts],
                self.points[ts + 1],
                self.points[ts + 2],
                self.points[ts + 3],
            )
            points.append(point)

        tmask = []
        for i in range(self.masks_chunk):
            ts = idx * self.masks_chunk + i
            tmask.append(self.masks[ts])

        mask = Mask(tmask)
        num_transform_vars = 9

        tv = []
        for i in range(num_transform_vars):
            ts = idx * num_transform_vars + i
            tv.append(self.transform_vars[ts])

        item = ItemSimulated(
            points, mask, VecRot(tv[0], tv[1], tv[2]), Trans(
                tv[3], tv[4], tv[5]), Stretch(tv[6], tv[7], tv[8]), self.sigma
        )
        return item

    # @profile
    def _create_points_mask(self):
        """Given the base points, perform dropout, spawn, noise and all the other
        messy functions, creating a new set of points. Internal function."""
        dropout_mask = array.array("d")
        points = array.array("d")

        for bp in self.gt_points:
            sx = 0.0
            sy = 0.0
            sz = 0.0
            tpoints = array.array("d")

            for i in range(self._max_spawn):

                if self.wobble != 0.0:
                    # sx = sy = sz = 0.0001
                    sx = random.gauss(0, self.wobble)
                    sy = random.gauss(0, self.wobble)
                    sz = random.gauss(0, self.wobble)

                # By organising the points as we do below, we get the correct
                # multiplication by matrices / tensors.
                tpoints.append(bp.x + sx)
                tpoints.append(bp.y + sy)
                tpoints.append(bp.z + sz)
                tpoints.append(1.0)

            if random.uniform(0, 1) >= self.dropout:
                for i in range(0, self._max_spawn):
                    if random.uniform(0, 1) < self.spawn:
                        dropout_mask.append(1.0)
                    else:
                        dropout_mask.append(0.0)
                    points.append(tpoints[i * 4])
                    points.append(tpoints[i * 4 + 1])
                    points.append(tpoints[i * 4 + 2])
                    points.append(tpoints[i * 4 + 3])
            else:
                # All dropped
                for i in range(self._max_spawn):
                    dropout_mask.append(0.0)
                    points.append(tpoints[i * 4])
                    points.append(tpoints[i * 4 + 1])
                    points.append(tpoints[i * 4 + 2])
                    points.append(tpoints[i * 4 + 3])

        return (points, dropout_mask)

    # @profile
    def _create_basic(self):
        """Create a set of rotations and set all the set sizes. Then call
        our threaded render to make the actual images. We render on demand at
        the moment, just creating the basics first. Internal function."""
        tx = 0
        ty = 0
        tz = 0

        sx = 1.0
        sy = 1.0
        sz = 1.0

        # Ensure an equal spread of data around all the rotation space so
        # we don't miss any particular areas
        rot = VecRot(0, 0, 0)

        for idx in tqdm(range(self.size), desc="Generating base data"):
            rot.random()
            tx = ((random.random() * 2.0) - 1.0) * self.max_trans
            ty = ((random.random() * 2.0) - 1.0) * self.max_trans
            tz = ((random.random() * 2.0) - 1.0) * self.max_trans

            if self.stretch:
                r0 = -1.0 + 2.0 * random.random()
                r1 = -1.0 + 2.0 * random.random()
                r2 = -1.0 + 2.0 * random.random()

                r0 = r0 / (1.0 + math.fabs(r0))
                r1 = r1 / (1.0 + math.fabs(r1))
                r2 = r2 / (1.0 + math.fabs(r2))
                
                sx = 1.0 + (r0 * self.max_stretch)
                sy = 1.0 + (r1 * self.max_stretch)
                sz = 1.0 + (r2 * self.max_stretch)

            points, dropout_mask = self._create_points_mask()

            if self.augment:
                tp = Points().from_chunk(points)
                new_points = rot.rotate_points(tp).get_chunk()

                for j in range(self.num_augment):
                    rot_a = VecRot(0, 0, math.pi * 2.0 * random.random())

                    # q0 = Quaternion(axis=rot.get_normalised(),
                    #                 radians=rot.get_length())
                    # q1 = Quaternion(axis=rot_a.get_normalised(),
                    #                 radians=rot_a.get_length())
                    # q2 = q0 * q1

                    #rot_f = VecRot(q2.axis[0] * q2.radians,
                    #               q2.axis[1] * q2.radians,
                    #               q2.axis[2] * q2.radians)

                    # What transformation do we really store here, as we have two!
                    # Our whole pipeline relies on there being one complete transform
                    # Composing the 3D base, then the 2D one doesn't work.
                    # To get what we want we modify the points by the initial rotation,
                    # keeping the extra augment till later.

                    self.transform_vars.append(rot_a.x)
                    self.transform_vars.append(rot_a.y)
                    self.transform_vars.append(rot_a.z)
                    self.transform_vars.append(tx)
                    self.transform_vars.append(ty)
                    self.transform_vars.append(tz)
                    self.transform_vars.append(sx)
                    self.transform_vars.append(sy)
                    self.transform_vars.append(sz)

                    self.points_chunk = len(new_points)
                    self.masks_chunk = len(dropout_mask)

                    for i in range(self.points_chunk):
                        self.points.append(new_points[i])
                    for i in range(self.masks_chunk):
                        self.masks.append(dropout_mask[i])

                    self.available.append(idx * self.num_augment + j)

            else:
                # Should always be the same
                self.transform_vars.append(rot.x)
                self.transform_vars.append(rot.y)
                self.transform_vars.append(rot.z)
                self.transform_vars.append(tx)
                self.transform_vars.append(ty)
                self.transform_vars.append(tz)
                self.transform_vars.append(sx)
                self.transform_vars.append(sy)
                self.transform_vars.append(sz)

                self.points_chunk = len(points)
                self.masks_chunk = len(dropout_mask)

                for i in range(self.points_chunk):
                    self.points.append(points[i])
                for i in range(self.masks_chunk):
                    self.masks.append(dropout_mask[i])

                del points[:]
                del dropout_mask[:]

                self.available.append(idx)

    def load(self, filename: str):
        """
        Load the data from a file instead of randomly creating them.

        Parameters
        ----------
        filename : str
           The path to the filename in question.

        Returns
        -------
        self
        """
        # Clear out first
        self.transform_vars = array.array("d")
        self.points = array.array("d")

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                (
                    self.size,
                    self.transform_vars,
                    self.points,
                    self.masks,
                    self.max_trans,
                    self.stretch,
                    self.max_stretch,
                    self.sigma,
                    self.augment,
                    self.num_augment,
                    self.dropout,
                    self.wobble,
                    self.spawn,
                    self._max_spawn
                ) = pickle.load(f)

        self.available = [i for i in range(0, self.size)]
        return self

    def save(self, filename):
        """
        Save the current loader to a file on disk. The file
        is saved using Python's pickle format.

        Parameters
        ----------
        filename : str
            The full path and filename to save to.

        Returns
        -------
        self
        """

        with open(filename, "wb") as f:
            pickle.dump(
                (
                    self.size,
                    self.transform_vars,
                    self.points,
                    self.masks,
                    self.max_trans,
                    self.stretch,
                    self.max_stretch,
                    self.sigma,
                    self.augment,
                    self.num_augment,
                    self.dropout,
                    self.wobble,
                    self.spawn,
                    self._max_spawn
                ),
                f,
                # pickle.HIGHEST_PROTOCOL,
            )
        return self

    def reserve(self, amount, alloc_csv=None):
        """
        A dataset can reserve an amount of data. This is randomly
        chosen by the dataloader, and returned as a large array or one
        can pass in a path to a file and choose that way, or it is
        passed in order to preserve any dataloader batches.

        Parameters
        ----------
        amount : int
            The amount requested by the dataset
        alloc_csv : str
            The path to a CSV file that determines the allocation.
            This is used when running the net deterministically.
            Default - None.
        Returns
        -------
        list
            The selected indexes of the items for the dataset.
        """

        if amount > self.remaining():
            raise ValueError(
                "Amount requested for reservation exceeds\
                amount of data remaining"
            )
        selected = []
        allocs = []
        removals = []

        if alloc_csv is not None:
            import csv

            with open(alloc_csv) as csvfile:
                csvallocs = csv.reader(csvfile)
                for row in csvallocs:
                    allocs = row

        for i in range(amount):
            idx = 0

            if len(allocs) > 0:
                idx = int(allocs[i])
                removals.append(idx)
                selected.append(self.available[idx])

            else:
                idx = random.randrange(len(self.available))
                selected.append(self.available[idx])
                del self.available[idx]

        if len(allocs) > 0:
            removals.sort(reverse=True)

            for r in removals:
                del self.available[r]

        return selected
