""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

renderer.py - our test functions using python's unittest.
This file tests the renderer

"""
import unittest

import torch
import random
import math
import util.plyobj as plyobj
from net.renderer import Splat
from util.math import TransTen, PointsTen, VecRot
from util.image import save_image


class Renderer(unittest.TestCase):

    def test_render(self):
        use_cuda = False
        device = torch.device("cuda" if use_cuda else "cpu")
        points = plyobj.load_obj("./objs/bunny_large.obj")
        base_points = points.to_ten(device=device)

        mask = []
        for _ in range(len(base_points)):
            mask.append(1.0)

        mask = torch.tensor(mask, device=device)
        xt = torch.tensor([0.0], dtype=torch.float32)
        yt = torch.tensor([0.0], dtype=torch.float32)
        zt = torch.tensor([0.0], dtype=torch.float32)

        splat = Splat(size=(32, 128, 128), device=device)

        r = VecRot(0, 0, 0).to_ten(device=device)
        r.random()
        t = TransTen(xt, yt, zt)

        result = splat.render(base_points, r, t, mask, sigma=2.2)
        self.assertTrue(torch.sum(result) > 200)
        save_image(torch.sum(result.detach(), dim=0), name="test_renderer_0.jpg")

    def test_dropout(self):
        use_cuda = False
        device = torch.device("cuda" if use_cuda else "cpu")
        base_points = PointsTen(device=device)
        base_points.from_points(plyobj.load_obj("./objs/bunny_large.obj"))
        mask = []

        for _ in range(len(base_points)):
            if random.uniform(0, 1) >= 0.5:
                mask.append(1.0)
            else:
                mask.append(0.0)

        mask = torch.tensor(mask, device=device)
        xt = torch.tensor([0.0], dtype=torch.float32)
        yt = torch.tensor([0.0], dtype=torch.float32)
        zt = torch.tensor([1.0], dtype=torch.float32)

        splat = Splat(math.radians(90), 1.0, 1.0, 10.0, size=(16, 32, 64), device=device)
        r = VecRot(0, math.radians(90), 0).to_ten(device=device)
        t = TransTen(xt, yt, zt)
        model = splat.render(base_points, r, t, mask, sigma=1.8)
        mask = []

        for _ in range(len(base_points)):
            mask.append(1.0)

        mask = torch.tensor(mask, device=device)
        model2 = splat.render(base_points, r, t, mask, sigma=1.8)

        self.assertTrue(torch.sum(model2) > torch.sum(model))
        save_image(torch.sum(model, dim=0), name="test_renderer_1.jpg")


if __name__ == "__main__":
    unittest.main()
