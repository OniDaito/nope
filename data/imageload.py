""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

imageload.py - the image loader for our network

"""

import os
from astropy.io import fits
import array
import torch
import csv
import pickle
from tqdm import tqdm
from util.math import Points, Point, gen_trans, gen_scale, PointsTen
from data.loader import Loader
from data.item import ItemImage, ItemImageClass, LoaderItem, ItemType


def reverse_graph(graph: Points, image_size):
    ''' The graph from the imageload .csv file comes in 3D image co-ordinates so it needs to be altered to fit.
    We are performing almost the reverse of the renderer.'''
    # For now, stick with a cube of -1 to 1 on each dimension
    assert(image_size[0] == image_size[1])
    dim = image_size[0]
    tgraph = PointsTen()
    tgraph.from_points(graph)
    scale = 2.0 / dim
    scale_mat = gen_scale(torch.tensor([scale]), torch.tensor([scale]), torch.tensor([scale]))
    trans_mat = gen_trans(torch.tensor([-1.0]), torch.tensor([-1.0]), torch.tensor([-1.0]))
    tgraph.data = torch.matmul(scale_mat, tgraph.data)
    tgraph.data = torch.matmul(trans_mat, tgraph.data)
    ngraph = tgraph.get_points()
    return ngraph


class ImageLoader(Loader):
    """A class that looks for images, saving the filepaths ready for
    use with the dataset class."""

    def __init__(self, size=1000, image_path=".", presigma=False, sigma=2.0, classes=False, class_suffix="", image_size=(200, 200)):
        """
        Create our ImageLoader.

        The image loader expects there to be either a directory matching the
        sigma passed in, or a flat directory (and presigma to be false).
        An example would be '/tmp/1.25', passing in
        '/tmp' as the image_path and 1.25 as sigma. None means there
        is just one directory with either a sigma of 1.0 or unknown sigmas.

        Parameters
        ----------
        size : int
            How big should this loader be? How many images do we want?
            Default: (1000)
        image_path : str
            The path to search for images.
        presigma : bool
            Are we expected preblurred in images in subdirs
        sigma : float
            The sigma of the images in question - default None
        classes: bool
            Are we importing the class masks as well
        class_suffix: str
            What differentiates a class-mask file from a data file?

        Returns
        -------
        self
        """

        # Total size of the data available and what we have allocated
        self.size = 0
        self.request_size = size
        self.counter = 0
        self.base_image_path = image_path
        self.available = array.array("L")
        self.data = []
        self.filenames = []
        self.deterministic = False
        self.sigma = sigma
        self.presigma = presigma
        self.classes = classes
        self.class_suffix = class_suffix
        self.graph = {}  # ID links to a bunch of points
        self.image_size = image_size

        print("Creating data from", self.base_image_path)

        if os.path.exists(self.base_image_path + "/log.csv"):
            with open(self.base_image_path + "/log.csv") as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',')
                for row in reader:
                    points = Points()
                    p0 = Point(float(row['p0x']), float(row['p0y']), float(row['p0z']), 1.0)
                    points.append(p0)
                    p1 = Point(float(row['p1x']), float(row['p1y']), float(row['p1z']), 1.0)
                    points.append(p1)
                    p2 = Point(float(row['p2x']), float(row['p2y']), float(row['p2z']), 1.0)
                    points.append(p2)
                    p3 = Point(float(row['p3x']), float(row['p3y']), float(row['p3z']), 1.0)
                    points.append(p3)
                    self.graph[row['id']] = points
        else:
            print("log.csv must exist along with the images.")
            assert(False)

        # Do we want pre-blurred images
        if self.presigma:
            self.sigmas = []
            subdirs = [x[0] for x in os.walk(self.base_image_path)]

            for s in subdirs:
                try:
                    sigma_level = float(s)
                    self.sigmas.append(sigma_level)
                except Exception as e:
                    print(e)

            self.sigmas.sort()
            self.sigmas.reverse()
        
        self._create_data()

    def _find_files(self, path, max_num=-1):
        """ Find the files from the path. Internal function."""
        img_files = []
        _, _, files = next(os.walk(path))

        if max_num == -1:
            max_num = len(files)
    
        pbar = tqdm(total=max_num)
        mini = 1e10
        maxi = 0

        for dirname, dirnames, filenames in os.walk(path):
            for i in range(len(filenames)):
                filename = filenames[i]
                img_extentions = ["fits", "FITS"]

                if any(x in filename for x in img_extentions):
                    # We need to check there are no duffers in this list
                    fpath = os.path.join(self.base_image_path, filename)

                    try:
                        with fits.open(fpath) as w:
                            hdul = w[0].data.byteswap().newbyteorder().astype('float32')
                            timg = torch.tensor(hdul, dtype=torch.float32, device="cpu")
                    
                            if torch.min(timg) < mini:
                                mini = torch.min(timg)
                            if torch.max(timg) > maxi:
                                maxi = torch.max(timg)

                            intensity = torch.sum(timg)
                            id_file = os.path.basename(filename)[:8]

                            if id_file in self.graph.keys():
                                if intensity > 0.0:
                                    pbar.update(1)
                                    img_files.append(fpath)
                    except:
                        print("Issue with FITS file", fpath)
                    
                    pbar.update(1)
                    img_files.append(fpath)
                    
                    if len(img_files) >= max_num:
                        pbar.close()
                        return img_files

        pbar.close()
        return img_files

    def set_sigma(self, sigma):
        """
        Set the sigma and create the data. We look for the sigma nearest
        the one that has been sent in. If it's different we recreate,
        otherwise we do nothing.
        
        Parameters
        ----------
        sigma : float
            The sigma to use.

        Returns
        -------
        self
        """
     
        if self.presigma:
            new_sigma = self.sigma

            for s in self.sigmas:
                if sigma - s > 0:
                    new_sigma = s

            if new_sigma != self.sigma:
                self.sigma = new_sigma
                self._create_data()
        else:
            # No need to recreate all the data. Lets just update the sigma on each datum
            self.sigma = sigma
            for datum in self.data:
                datum.sigma = self.sigma

        return self

    def _create_data(self):
        """
        We look for directories fitting the path ceppath + "/<sigma>/"
        Couple of choices here
        Internal function.
        """
        path = self.base_image_path
        self.data = []

        if self.sigma is not None and self.presigma:
            path = self.base_image_path + "/" + str(int(self.sigma)).zfill(2)
            path1 = self.base_image_path + "/" + str(int(self.sigma))
            path2 = self.base_image_path + "/" + str(self.sigma)

            if os.path.exists(path1):
                path = path1
            if os.path.exists(path2):
                path = path2

        if len(self.filenames) == 0:
            self.filenames = self._find_files(path, -1)

        if self.classes:
            classes = []
            images = []

            for name in self.filenames:
                if self.class_suffix in name:
                    classes.append(name)
                else:
                    images.append(name)

            classes.sort()
            images.sort()
            assert(len(classes) == len(images))

            for i, _ in enumerate(classes):
                file_id = os.path.basename(images[i])[:8]
                graph = self.graph[file_id]
                # TODO - should we move reversed?
                # reversed = reverse_graph(graph, self.image_size)
                self.data.append(ItemImageClass(images[i], classes[i], graph, self.sigma))

        else:
            for name in self.filenames:
                if 'layered' in name:
                    file_id = os.path.basename(name)[:8]
                    graph = self.graph[file_id]
                    # TODO - should we move reversed?
                    #reversed = reverse_graph(graph, self.image_size)
                    self.data.append(ItemImage(name, graph, self.sigma))

        if len(self.data) > self.request_size:
            self.data = self.data[:self.request_size]

        self.size = len(self.data)
        self.available = list(range(self.size))

    def remaining(self) -> int:
        """
        Return the number of data remaining that haven't
        been claimed by the dataset.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of remaining items
        """
        return len(self.available)

    def __next__(self):

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

    def __getitem__(self, idx) -> LoaderItem:
        return self.data[idx]

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

        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                (
                    self.size,
                    self.data,
                    self.base_image_path,
                    self.deterministic,
                    self.sigma,
                ) = pickle.load(f)

                print("Loaded imageloader from", filename)

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
                    self.data,
                    self.base_image_path,
                    self.deterministic,
                    self.sigma,
                ),
                f,
                pickle.HIGHEST_PROTOCOL,
            )
        return self
