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
from util.plyobj import load_obj, load_ply, save_ply


class Model(object):
    '''
    A basic class that holds points, PointsTen and a set
    of indices that group these points. This forms the
    structure we are trying to come up with that represents
    our worm neurons. The points are classified 0 to 3 depending
    on which neuron they belong to.
    '''

    def __init__(self):
        self.points = Points()
        self.indices = []

    def add_points(self, points: Points, idp: int):
        for p in points:
            self.points.append(p)
        
        for _ in range(len(points)):
            self.indices.append(idp)

    def make_ten(self, device="cpu") -> Tuple[PointsTen, list]:
        ''' 
        Create the tensor we will optimise now we've added all the points.
        '''
        self.data = self.points.to_ten(device=device)
        return (self.data, self.indices)

    def from_ten(self, points: PointsTen):
        # In case it's just the single model
        if len(self.indices) == 1:
            self.points.append(points.get_points())   
            return 

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

    def save_ply(self, path="model.ply"):
        vertices = []
        tv = self.data.data.clone().cpu().detach().numpy()

        for v in tv:
            vertices.append((v[0][0], v[1][0], v[2][0], 1.0))
        save_ply(path, vertices, self.indices)

    def from_csv(self, csv_path):
        raise ValueError("Not implemented yet")

    def save_csv(self, csv_path):
        raise ValueError("Not implemented yet")
