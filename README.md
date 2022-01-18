# Neuron Orientation / Pose Estimation (NOPE)

An adaptation of HOLLy to the problem of identifying certain neurons of interest in the organism *C. elegans*. It works by taking a known model of said neurons and aligning them to best fit the data.

## Example run

    python ./train.py --savedir ../runs/nope_2021_10_25 --save-interval 800 --train-size 40000 --test-size 600 --valid-size 20 --objpath ./objs/bunny_large.obj --buffer-size 2 --epochs 20 --batch-size 2 --cont --image-width 128 --image-height 64 --image-depth 64 --log-interval 100 --num-points 350 --lr 0.0004
