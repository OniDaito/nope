"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

model.py - a collection of PointsTen objects that represent
our model of C. elegans

"""
from typing import Tuple
import torch
import random
from util.math import Points, PointsTen
from util.plyobj import load_obj, load_ply


class Model(object):

    def __init__(self):
        self.points = []
        self.indices = []

    def add_points(self, points: Points, idp: int):
        x = [p[0] for p in self.points]
  
        if idp in x:
            raise ValueError("Cannot save points with id of", idp)

        self.points.append((idp, points))
        self._sort_points()
        self.indices = []
        idx = 0

        for tp in self.points:
            self.indices.append(idx)
            idx += len(tp[1])

    def _sort_points(self):
        self.points.sort(key=lambda x: x[0])

    def get_ten(self, device="cpu") -> tuple(PointsTen, list):
        tpoints = Points()

        for tp in self.points:
            tpoints.cat(tp[1])

        return (tpoints.to_ten(device=device), self.indices)

    def from_ten(self, points: PointsTen):
        tensors = points.data.split(self.indices)
        for t in tensors:
            tpoints = PointsTen(device=points.device)
            tpoints.from_tensor(t)
            self.points.append(tpoints.get_points())     

    def load_models(self, paths):
        order = 0
        for path in paths:
            if 'ply' in path:
                self.add_points(load_ply(path), order)
                order += 1
            elif 'obj' in path:
                self.add_points(load_obj(path), order)
                order += 1
            else:
                raise ValueError("Path must be to a ply or obj.")

    def from_csv(self, csv_path):
        raise ValueError("Not implemented yet")

    def save_csv(self, csv_path):
        raise ValueError("Not implemented yet")
