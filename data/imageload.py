""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

imageload.py - the image loader for our network

"""

import os
from astropy.io import fits
import array
import torch
from tqdm import tqdm
from data.loader import Loader, LoaderItem, ItemType


class ItemImage(LoaderItem):
    def __init__(self, path, sigma: float):
        self.type = ItemType.FITSIMAGE
        self.path = path
        self.sigma = sigma

    def unpack(self):
        return self.path


class ImageLoader(Loader):
    """A class that looks for images, saving the filepaths ready for
    use with the dataset class."""

    def __init__(self, size=1000, image_path=".", sigma=0):
        """
        Create our ImageLoader.

        Parameters
        ----------
        size : int
            How big should this loader be? How many images do we want?
            Default: (1000)
        image_path : str
            The path to search for images.
       
        Returns
        -------
        self
        """

        # Total size of the data available and what we have allocated
        self.size = size
        self.counter = 0
        self.base_image_path = image_path
        self.available = array.array("L")
        self.filenames = []
        self.deterministic = False
        self.sigma = sigma

        self._create_data()

    def _find_files(self, path, max_num):
        """ Find the files from the path. Internal function."""
        img_files = []
        pbar = tqdm(total=max_num)
        idx = 0

        for dirname, dirnames, filenames in os.walk(self.base_image_path):
            for i in range(len(filenames)):
                filename = filenames[i]
                img_extentions = ["fits", "FITS"]

                if any(x in filename for x in img_extentions):
                    # We need to check there are no duffers in this list
                    if "_layered" in filename:
                        fpath = os.path.join(path, filename)

                        with fits.open(fpath) as w:
                            hdul = w[0].data.byteswap().newbyteorder().astype('float32')
                            timg = torch.tensor(hdul, dtype=torch.float32)
                            print("SHS", timg.shape)
                            assert(len(timg.shape) == 3)

                            pbar.update(1)
                            img_files.append(fpath)
                            self.available.append(idx)
                            idx += 1

                        if len(img_files) >= max_num:
                            pbar.close()
                            return img_files

        pbar.close()
        print("Found", len(img_files), "fits files for training set.")
        return img_files

    def _create_data(self):
        """
        Internal function.
        """
        path = self.base_image_path

        print("Creating data from", path)
        self.filenames = self._find_files(path, self.size)
        assert len(self.filenames) == self.size

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

    def set_sigma(self, sigma):
        self.sigma = sigma

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
        return ItemImage(self.filenames[idx], self.sigma)
