"""  # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/       # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/       # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

util_points.py - functions to do with our points tensor
such as loading it from disk or initialising it with 
random data.
"""

import random
from util.math import Points, Point, PointsTen


def classify_kmeans(points: Points, num_classes=4):
    ''' Given a set of points, place each point into a class based on the
    K-Means clustering algorithm. '''
    from scipy.cluster.vq import kmeans2

    #whitened = whiten(points.get_iterable())
    #code_book, distortion = kmeans(whitened, num_classes)
    code_book, labels = kmeans2(points.get_iterable(), 4, minit="++")

    # Now add each point to a class based on distance
    centroids = []

    for c in range(num_classes):
        cb = code_book[c]
        cp = Point(cb[0], cb[1], cb[2], 1.0)
        centroids.append(cp)

    print("Centroids:\n" + str(centroids[0]) + "\n" + str(centroids[1]) + "\n" + str(centroids[2]) + "\n" + str(centroids[3]))

    point_classes = []

    for p in points:
        d = 100.0
        pclass = 0

        for ic, c in enumerate(centroids):
            dd = c.dist(p)

            if dd < d:
                d = dd
                pclass = ic
  
        point_classes.append(pclass)

    return point_classes


def load_points(filename) -> Points:
    """
    Load the points from a text file.

    Parameters
    ----------
    filename : str
        A path and filename for the points file

    Returns
    -------
    Points
        Our Points instance

    """
    points = Points(size=0)
    i = 0

    with open(filename, "r") as f:
        for line in f.readlines():
            comment = "".join(line.split())
            if comment[0] != "#":
                tokens = line.replace("\n", "").split(",")
                x = float(tokens[0])
                y = float(tokens[1])
                z = float(tokens[2])
                points.append(Point(x, y, z))
                i = i + 1

    return points


def init_points(num_points=500, device="cpu", deterministic=False, spread=1.0) -> Points:
    """
    Rather than load a torus or fixed shape, create a
    tensor that contains a random number of points.

    Parameters
    ----------
    num_points : int
        The number of points to make (default 500).
    device : str
        The device that holds the points (cuda / cpu).
        Default - cpu.
    deterministic : bool
        Are we going for a deterministic run?
        Default - False.
    spread: float
        How wide should the area where points are placed be?
        Default - 1.0

    Returns
    -------
    PointsTen
        Our Points in PointsTen form.
    """
    points = Points()
    if deterministic:
        # TODO - can we guarantee this gives the same numbers?
        random.seed(a=9001)

    # Everything is roughly centred in the images so spawn
    # the points close to the centre
    for i in range(0, num_points):
        p = Point(
            random.uniform(-spread/2, spread/2),
            random.uniform(-spread/2, spread/2),
            random.uniform(-spread/2, spread/2),
            1.0,
        )
        points.append(p)

    return points


def save_points(filename, points: Points):
    """
    Save the points to a text file.

    Parameters
    ----------
    filename : str
        The file path to save to.
    points : Points
        The points to save.

    Returns
    -------
    None
    """
    tt = points.data.cpu().detach().numpy()
    with open(filename, "w") as f:
        for i in range(0, len(points.data)):
            x = tt[i][0][0]
            y = tt[i][1][0]
            z = tt[i][2][0]
            f.write(str(x) + "," + str(y) + "," + str(z) + "\n")
