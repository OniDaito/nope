import os
import random
import pickle
import array
import math
from tqdm import tqdm
from enum import Enum
from util.math import Points, Point, Mask, Trans, VecRot, Stretch


ItemType = Enum("SetType", "SIMULATED FITSIMAGE")

class LoaderItem:
    """The item returned by any of the various Loaders.
    This is the base class, expanded upon below."""

    def __init__(self):
        self.type = ItemType.SIMULATED

    def unpack(self):
        assert False
        return []


class ItemSimulated(LoaderItem):
    """ The Simulated items returned by the basic loader."""

    def __init__(
        self, points: Points, mask: Mask, angle_axis: VecRot, trans: Trans, stretch: Stretch, sigma: float
    ):
        """
        Create our ItemSimulated.

        Parameters
        ----------
        points : Points
           The points that make up this datum.
        mask : Mask
            The mask for the points.
        angle_axis : VecRot
            The rotation of this datum.
        trans : Trans
            The translation of this datum.
        sigma : float
            The sigma this datum should be rendered with.

        Returns
        -------
        self
        """
        super().__init__()
        self.type = ItemType.SIMULATED
        self.points = points
        self.mask = mask
        self.angle_axis = angle_axis
        self.trans = trans
        self.stretch = stretch
        self.sigma = sigma

    def unpack(self) -> tuple:
        """
        Unpack the item, return a tuple.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            The item as a tuple in the following order:
            points, mask, rotation, translation, sigma
        """
        return (self.points, self.mask, self.angle_axis, self.trans, self.sigma)


class ItemImage():
    def __init__(self, path, graph: Points, sigma=1.0):
        #self.type = ItemType.FITSIMAGE
        self.path = path
        self.graph = graph
        self.sigma = sigma

    def unpack(self):
        return self.path


class ItemImageClass():
    def __init__(self, path, class_path, graph: Points, sigma=1.0):
        #self.type = ItemType.FITSPLUSCLASS
        self.path = path
        self.class_path = class_path
        self.graph = graph
        self.sigma = sigma

    def unpack(self):
        return self.path


