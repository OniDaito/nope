"""
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/
Author : Benjamin Blundell - benjamin.blundell@kcl.ac.uk

viz_elegans.py - Visualise the groups and results.


"""

import argparse
import os
import torch
from elegans.funcs import load_saved_model
from util.image import load_fits, save_fits, save_image
from util.math import TransTen3D
from util.points import classify_kmeans
import torch.nn.functional as F
from net.renderer3d import Splat as Splat3D
import numpy as np


def viz(occupancy: np.array, ogvolume: np.array):
    from vedo import Volume, show

    data_matrix0 = occupancy[:, :, :, 0].squeeze()
    data_matrix1 = occupancy[:, :, :, 1].squeeze()
    data_matrix2 = occupancy[:, :, :, 2].squeeze()
    data_matrix3 = occupancy[:, :, :, 3].squeeze()

    vol0 = Volume(data_matrix0, c='Purples', mapper='gpu')
    vol1 = Volume(data_matrix1, c='Greens', mapper='gpu')
    vol2 = Volume(data_matrix2, c='Blues', mapper='gpu')
    vol3 = Volume(data_matrix3, c='Reds', mapper='gpu')
    volog = Volume(ogvolume, c='Greys', mapper='gpu')

    show(vol0, vol1, vol2, vol3, volog, axes=1).close()


def image_test(model, points, device, input_image, source_volume, normaliser, groups):
    """Test our model by loading an image and seeing how well we
    can match it. We might need to duplicate to match the batch size."""

    # Need to call model.eval() to set certain layers to eval mode. Ones
    # like dropout and what not
    with torch.no_grad():
        model.eval()
        dimmed = torch.unsqueeze(input_image, dim=0)  # expand to 3 dims
        batch = torch.unsqueeze(dimmed, dim=0)
        print("Input Shape", batch.shape)
        normed = normaliser.normalise(batch)
        normed = normed.to(device)
        x = normaliser.normalise(model.forward(normed, points))
        x = torch.squeeze(x)
        im = torch.squeeze(normed)
        print("Output Shape", x.shape)

        loss = F.l1_loss(x, im, reduction="sum")
        print("loss", float(loss.item()))

        # Save the results as images
        if os.path.exists("guess.fits"):
            os.remove("guess.fits")

        save_fits(x, name="guess.fits", flip=True)
        save_image(x, name="guess.jpg")

        # Now we have a guess, lets get the params and points
        (rot, trans, stretch, sigma) = model.final_params(model._final[0])
        print("Predicted params (rot, trans, stretch, sigma): ", rot, str(trans), str(stretch), str(sigma))

        # Create an occupancy volume for our final predictions
        # This will match the source volume we have
        vd = source_volume.shape[0]
        vh = source_volume.shape[1]
        vw = source_volume.shape[2]
        print("Shape", source_volume.shape)

        occupancy = np.zeros((vd, vh, vw, 4), dtype=np.float32)
        splat3d = Splat3D(size=(vd, vh, vw),  device=device)
        trans3d = TransTen3D(trans.x, trans.y, torch.tensor([0], dtype=torch.float32, device=device))
        mask = points.data.new_full([points.data.shape[0], 1, 1], fill_value=1.0)
        res = splat3d.render_rot_mat(points, rot, trans3d, stretch, mask, sigma)
        save_fits(res, name="guess_og.fits", flip=True)

        print("Points shape", points.data.shape)

        # Render each group out to our occupancy list and calculate the score against the original volume
        for gid in range(4):
            gmask = points.data.new_full([points.data.shape[0], 1, 1], fill_value=0.0)

            for ip, g in enumerate(groups):
                if g == gid:
                    gmask.data[ip][0][0] = 1.0

            res = splat3d.render_rot_mat(points, rot, trans3d, stretch, gmask, sigma)
            occupancy[:, :, :, gid] = np.flip(res.numpy(), axis=1)
            save_fits(occupancy[:, :, :, gid], name="guess3d_" + str(gid) + ".fits")
            #gscore = np.sum(np.multiply(occupancy[:, :, :, gid], source_volume))
            #print("Group:", gid, "# points:", torch.sum(gmask), "score", gscore)

        print("Max score", np.max(occupancy))

    return occupancy


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="HOLLy C. Elegans prog.")

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="random seed (default: 1)"
    )
    parser.add_argument(
        "--sigma", type=float, default=1.8, help="Sigma to use"
    )
    parser.add_argument(
        "--width", type=int, default=200, help="Image width"
    )
    parser.add_argument(
        "--height", type=int, default=200, help="Image height"
    )
    parser.add_argument(
        "--roix", type=int, default=200, help="ROI X"
    )
    parser.add_argument(
        "--roiy", type=int, default=40, help="ROI Y"
    )
    parser.add_argument(
        "--roiw", type=int, default=200, help="ROI dimension"
    )
    parser.add_argument(
        "--load", default="./save", help="The name for checkpoint save directory."
    )
    parser.add_argument('--base', default="")
    parser.add_argument('--rep', default="")
    parser.add_argument(
        "--input-image", default="", help="The 2D image from which to base the prediction on."
    )
    parser.add_argument(
        "--input-volume", default="", help="The input volume - the original source."
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint.pth.tar",
        help="The name for checkpoint save file.",
    )

    # Initial setup of PyTorch
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Load all the base data
    model, points, points_tensor, device, normaliser = load_saved_model(args, args.height, args.width, device)
    base_points = points.get_points()
    groups = classify_kmeans(base_points)

    
    input_image = load_fits(args.input_image)
    source_volume = load_fits(args.input_volume)
    x = args.roix
    y = args.roiy
    w = args.roiw 
    source_volume = source_volume[:, y:y+w, x:x+w]

    occupancy = image_test(model, points, device, input_image, source_volume, normaliser, groups)
    viz(occupancy, source_volume)
    
