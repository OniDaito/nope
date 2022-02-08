""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

batcher.py - tyrns single datum access into a set of data,
also known as a batch. This sits atop something that
supports iteration, typically a buffer.

"""

import torch
from data.buffer import ItemRendered
from globals import DTYPE


class Batch(object):
    ''' A little dictionary of sorts that holds the actual data we need 
    for the neural net (the images) and the associated data used to make
    these images.'''
    
    def __init__(self, batch_size: int, isize, device):
        self._idx = 0
        
        self.data = torch.zeros(
            (batch_size, 1, isize[0], isize[1], isize[2]),
            dtype=DTYPE,
            device=device,
        )

        self.rotations = []
        self.translations = []
        self.sigmas = []

    def add_datum(self, datum: ItemRendered):
        self.data[self._idx][0] = datum.datum
        self.rotations.append(datum.rotation)
        self.translations.append(datum.translation)
        self.sigmas.append(datum.sigma)
        self._idx += 1


class Batcher:
    def __init__(self, buffer, batch_size=16):
        """
        Create our batcher

        Parameters
        ----------
        buffer : Buffer
            The buffer behind the batcher
        batch_size : int
            How big is the batch?

        Returns
        -------
        Batcher
        """
        self.batch_size = batch_size
        self.buffer = buffer

    def __iter__(self):
        return self

    def __len__(self):
        """ Return the number of batches."""
        return int(len(self.buffer) / self.batch_size)

    def __next__(self) -> Batch:
        """
        Return a batch suitable for training.

        This function effectively regroups the ItemRendered into 5 tensors of length batch_size
        rather than a list of batch_size ItemRendereds
        """

        try:
            batch = Batch(self.batch_size, self.buffer.image_size(), self.buffer.device)
            for i in range(self.batch_size):
                datum = self.buffer.__next__()
                batch.add_datum(datum)
            return batch

        except StopIteration:
            raise StopIteration("Batcher reached the end of the dataset.")

        except Exception as e:
            raise e
