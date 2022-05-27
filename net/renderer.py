""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/          # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/          # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

renderer.py - Perform splatting of gaussians with torch
functions. Based on the DirectX graphics pipeline.
"""

import torch
import math
import torch.nn.functional as F

from util.math import (
    gen_mat_from_rod,
    gen_ortho,
    gen_trans_xyz,
    gen_identity,
    gen_screen,
    gen_scale,
    VecRotTen,
    TransTen,
    PointsTen,
    make_gaussian_kernel,
)
from globals import DTYPE


class Splat(object):
    """Our splatter class that generates matrices, loads 3D
    points and spits them out to a 2D image with gaussian
    blobs. The gaussians are computed in screen/pixel space
    with everything else following the DirectX style
    pipeline."""

    # TODO - we should really check where requires grad is actually needed.

    def __init__(
        self, size=(25, 150, 320), device=torch.device("cpu")
    ):
        """
        Initialise the renderer.

        Parameters
        ----------
     
        size : tuple
            The size of the rendered image, in pixels  z, y and x dimensions (default: (25, 150, 320))

        Returns
        -------
        Splat
            The renderer itself

        """

        self.size = size
        # self.near = near
        # self.far = far
        self.device = device
        # self.perspective = gen_perspective(fov, aspect, near, far)
        self.ortho = gen_ortho(self.size, device)
        # self.modelview = gen_identity(device=self.device)
        self.trans_mat = gen_identity(device=self.device)
        self.rot_mat = gen_identity(device=self.device)
        self.scale_mat = gen_scale(
            torch.tensor([0.5], dtype=DTYPE, device=self.device),
            torch.tensor([0.5], dtype=DTYPE, device=self.device),
            torch.tensor([0.5], dtype=DTYPE, device=self.device),
        )
        self.z_correct_mat = gen_scale(
            torch.tensor([1.0], dtype=DTYPE, device=self.device),
            torch.tensor([1.0], dtype=DTYPE, device=self.device),
            torch.tensor([size[0] / size[1]], dtype=DTYPE, device=self.device),
        )

        self.ndc = gen_screen(self.size, device=self.device)
        self.xs = torch.tensor([0], dtype=DTYPE)
        self.ys = torch.tensor([0], dtype=DTYPE)
        self.zs = torch.tensor([0], dtype=DTYPE)

        mask = []
        for _ in range(0, 200):
            mask.append(1.0)
        self.mask = torch.tensor(mask, device=self.device)

    def _gen_mats(self, points: PointsTen):
        """
        Internal function.
        Generate the tensors we need to do the rendering.
        These are support matrices needed by pytorch in order
        to convert out points to 2D ones all in the same
        final tensor."""

        # X indices
        numbers = list(range(0, self.size[2]))
        rectangle = [numbers for x in range(0, self.size[1])]
        cuboid = []
        hypercube = []

        for i in range(0, self.size[0]):
            cuboid.append(rectangle)

        for i in range(0, points.data.shape[0]):
            hypercube.append(cuboid)

        self.xs = torch.tensor(hypercube, dtype=DTYPE, device=self.device)

        # The Y indices
        rectangle = []
        cuboid = []
        hypercube = []

        for i in range(0, self.size[1]):
            numbers = [i for x in range(self.size[2])]
            rectangle.append(numbers)

        for i in range(0, self.size[0]):
            cuboid.append(rectangle)

        for i in range(0, points.data.shape[0]):
            hypercube.append(cuboid)

        self.ys = torch.tensor(hypercube, dtype=DTYPE, device=self.device)

        # The Z indices
        rectangle = []
        cuboid = []
        hypercube = []

        for i in range(0, self.size[0]):
            for j in range(0, self.size[1]):
                numbers = [i for x in range(0, self.size[2])]
                rectangle.append(numbers)
            cuboid.append(rectangle)
            rectangle = []

        for i in range(0, points.data.shape[0]):
            hypercube.append(cuboid)

        self.zs = torch.tensor(hypercube, dtype=DTYPE, device=self.device)

    def transform_points(
        self, points: torch.Tensor, a: VecRotTen, t: TransTen
    ) -> torch.Tensor:
        """
        Transform points with translation and rotation. A utility
        function used in eval.py to produce a list of transformed points
        we can save to a file.

        Parameters
        ----------
        points : torch.Tensor
            The points all converted to a tensor.

        a : VecRotTen
            The rotation to apply to these points.

        t : TransTen
            The translation to apply to these points.

        Returns
        -------
        torch.Tensor
            The converted points as a tensor.

        """
        self.rot_mat = gen_mat_from_rod(a)
        self.trans_mat = gen_trans_xyz(t.x, t.y, t.z)
        self.modelview = torch.matmul(self.rot_mat, self.trans_mat)
        o = torch.matmul(self.modelview, points.data)
        return o

    def to(self, device):
        """
        Move this class and all it's associated data from
        one device to another.

        Parameters
        ----------
        device : str
            The device we are moving the renderer to - CUDA or cpu.

        Returns
        -------
        Splat
            The renderer itself.

        """
        self.device = torch.device(device)
        # self.perspective = self.perspective.to(device)
        # self.modelview = self.modelview.to(device)
        self.trans_mat = self.trans_mat.to(device)
        self.rot_mat = self.rot_mat.to(device)
        self.scale_mat = self.scale_mat.to(device)
        self.ndc = self.ndc.to(device)
        self.ortho = self.ortho.to(device)
        self.xs = self.xs.to(device)
        self.ys = self.ys.to(device)
        self.zs = self.zs.to(device)
        self.z_correct_mat = self.z_correct_mat.to(device)
        # self.w_mask = self.w_mask.to(device)
        return self

    def render(
        self,
        points: PointsTen,
        rot: VecRotTen,
        trans: TransTen,
        mask: torch.Tensor,
        sigma=1.25,
    ):
        """
        Generate an image. We take the points, a mask, an output filename
        and 2 classed that represent the rodrigues vector and the translation.
        Sigma refers to the spread of the gaussian. The mask is used to ignore
        some of the points if necessary.

        Parameters
        ----------
        points : PointsTen
            The points we are predicting.
        rot : VecRotTen
            The rotation as a vector
        trans : TransTen
            The translation of the points.
        mask : torch.Tensor
            A series of 1.0s or 0.0s to mask out certain points.
        sigma : float
            The sigma value to render our image with.

        Returns
        -------
        None

        """

        assert mask is not None
        if self.xs.shape[0] != points.data.shape[0]:
            self._gen_mats(points)

        # This section causes upto a 20% hit on the GPU perf
        self.rot_mat = gen_mat_from_rod(rot)
        self.trans_mat = gen_trans_xyz(trans.x, trans.y, trans.z)
        p0 = torch.matmul(self.scale_mat, points.data)
        p1 = torch.matmul(self.rot_mat, p0)
        p2 = torch.matmul(self.trans_mat, p1)
        p3 = torch.matmul(self.z_correct_mat, p2)
        p4 = torch.matmul(self.ortho, p3)
        s = torch.matmul(self.ndc, p4)

        px = s.narrow(1, 0, 1).reshape(len(points), 1, 1, 1)
        py = s.narrow(1, 1, 1).reshape(len(points), 1, 1, 1)
        pz = s.narrow(1, 2, 1).reshape(len(points), 1, 1, 1)
        ex = px.expand(points.data.shape[0], self.size[0], self.size[1], self.size[2])
        ey = py.expand(points.data.shape[0], self.size[0], self.size[1], self.size[2])
        ez = pz.expand(points.data.shape[0], self.size[0], self.size[1], self.size[2])

        # Expand the mask out so we can cancel out the contribution
        # of some of the points
        mask = mask.reshape(mask.shape[0], 1, 1, 1)
        mask = mask.expand(mask.shape[0], ey.shape[1], ey.shape[2], ey.shape[3])

        model = (
            1.0
            / (2.0 * math.pi * sigma ** 3)
            * torch.sum(
                mask
                * torch.exp(
                    -((ex - self.xs) ** 2 + (ey - self.ys) ** 2 + (ez - self.zs) ** 2)
                    / (2 * sigma ** 2)
                ),
                dim=0,
            )
        )

        return model

    def render_conv(
        self,
        points: PointsTen,
        rot: VecRotTen,
        trans: TransTen,
        mask: torch.Tensor,
        sigma=1.25,
    ):
        """
        Generate an image. We take the points, a mask, an output filename
        and 2 classed that represent the rodrigues vector and the translation.
        Sigma refers to the spread of the gaussian. The mask is used to ignore
        some of the points if necessary.

        Parameters
        ----------
        points : PointsTen
            The points we are predicting.
        rot : VecRotTen
            The rotation as a vector
        trans : TransTen
            The translation of the points.
        mask : torch.Tensor
            A series of 1.0s or 0.0s to mask out certain points.
        sigma : float
            The sigma value to render our image with.

        Returns
        -------
        None

        """

        assert mask is not None

        gkern = make_gaussian_kernel(sigma).to(dtype=DTYPE, device=self.device)

        if self.xs.shape[0] != points.data.shape[0]:
            self._gen_mats(points)

        # This section causes upto a 20% hit on the GPU perf
        self.rot_mat = gen_mat_from_rod(rot)
        self.trans_mat = gen_trans_xyz(trans.x, trans.y, trans.z)
        self.rot_mat = self.rot_mat.to(dtype=DTYPE)
        self.modelview = torch.matmul(
            torch.matmul(self.scale_mat, self.rot_mat), self.trans_mat
        )
        o = torch.matmul(self.modelview, points.data)
        q = torch.matmul(self.z_correct_mat, points.data)
        s = torch.matmul(self.ndc, q)
        px = s.narrow(1, 0, 1).reshape(len(points), 1, 1, 1)
        py = s.narrow(1, 1, 1).reshape(len(points), 1, 1, 1)
        pz = s.narrow(1, 2, 1).reshape(len(points), 1, 1, 1)
        ex = px.expand(points.data.shape[0], self.size[0], self.size[1], self.size[2])
        ey = py.expand(points.data.shape[0], self.size[0], self.size[1], self.size[2])
        ez = pz.expand(points.data.shape[0], self.size[0], self.size[1], self.size[2])

        # Expand the mask out so we can cancel out the contribution
        # of some of the points
        mask = mask.reshape(mask.shape[0], 1, 1, 1)
        mask = mask.expand(mask.shape[0], ey.shape[1], ey.shape[2], ey.shape[3])

        # New method using a kernel. We create a 3D cube with the points in it, then convolve
        # No sub pixel accuracy due to rounding but hey
        rounded = torch.round(s).squeeze().narrow(1, 0, 3)
        px = rounded.narrow(1, 0, 1).reshape(len(points), 1, 1, 1)
        py = rounded.narrow(1, 1, 1).reshape(len(points), 1, 1, 1)
        pz = rounded.narrow(1, 2, 1).reshape(len(points), 1, 1, 1)
        ex = px.expand(points.data.shape[0], self.size[0], self.size[1], self.size[2])
        ey = py.expand(points.data.shape[0], self.size[0], self.size[1], self.size[2])
        ez = pz.expand(points.data.shape[0], self.size[0], self.size[1], self.size[2])

        pcube_x = ex == self.xs
        pcube_y = ey == self.ys
        pcube_z = ez == self.zs

        pcube = torch.logical_and(pcube_x, pcube_y)
        pcube = torch.logical_and(pcube, pcube_z).to(dtype=DTYPE).sum(dim=0)
        vol_in = pcube.reshape(1, 1, *pcube.shape)

        k3d = torch.einsum("i,j,k->ijk", gkern, gkern, gkern)
        k3d = k3d / k3d.sum()
        k3d = k3d.to(dtype=DTYPE)
        # print("Shapes", pcube.shape, vol_in.shape, k3d.shape)
        vol_3d = F.conv3d(
            vol_in, k3d.reshape(1, 1, *k3d.shape), stride=1, padding=len(gkern) // 2
        )
        model = vol_3d.reshape(self.size)

        return model
