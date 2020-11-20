# Unsupervised-Sketch-to-Photo-Synthesis
This is the official released code for our [paper](https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2020sketchECCV.pdf), Unsupervised Sketch to Photo Synthesis, which has been accepted as an Oral paper of ECCV 2020. Please find further details in the project [page](http://sketch.icsi.berkeley.edu/).

	@article{liu2020unsupervised,
	  author     = {Runtao Liu and
                    Qian Yu and
                    Stella Yu},
	  title      = {Unsupervised Sketch to Photo Synthesis},
	  conference = {ECCV 2020},
	  year       = {2020}
	}

## Environment: 

Please install Pytorch 1.4.0 and other requirements like visdom.

## Dataset and Released Model

We prepared an example dataset of the shoes and it can be downloaded [here](https://drive.google.com/file/d/1ybxtl_843n8uMCgfuVE6PImD40GjX6-R/view?usp=sharing). 

We also released our trained model checkpoint [here](https://drive.google.com/file/d/1fh81dYmhnodqhk_eCA8njAH1rLXIJw_V/view?usp=sharing).

## Testing with Released Model

Testing command for step 1:

    python -u test.py \
        --only_fakephoto \
        --sp_attention 1 \
        --load_size 128 \
        --crop_size 128 \
        --results_dir ./results/step1_release_model/ \
        --input_nc 1 \
        --output_nc 1 \
        --epoch 480 \
        --dataroot ../datasets/shoes \
        --name shoes \
        --checkpoints_dir ../experiments/step1/step1_release_model/model/

After the testing of step 1, the output results should be placed at the path datasets/shoes\_step2/testA.

Testing command for step 2:

    python -u test.py \
        --only_fakephoto \
        --netG resnet_12blocks \
        --dataroot ../datasets/shoes_step2 \
        --load_size 128 \
        --crop_size 128 \
        --results_dir ./results/step2_release_model/ \
        --input_nc 3 \
        --output_nc 3 \
        --epoch 290 \
        --name shoes \
        --checkpoints_dir ../experiments/step2/step2_release_model/model/

We has updated the checkpoint whose performance is slightly better than that reported in the paper. The FID values on the shoe dataset of released models are:

step1: 44.35 

step2: 48.20 (report: 48.73)

Note that the FID of step1 should be calculated between the output and **grayscale** images with 128x128 resolution.
The FID of step2, which is the final FID result, should be calculated between the output and **RGB** images with 128x128 resolution.

## Training

Step 1 training command: 

    python train.py \
        --batch_size 1 \
        --sp_attention 1 \
        --display_freq 20 \
        --prob_add_noise 0.2 \
        --prob_add_patch 0.3 \
        --load_size 140 \
        --crop_size 128 \
        --input_nc 1 \
        --output_nc 1 \
        --niter 100 \
        --niter_decay 600 \
        --dataroot ../datasets/shoes \
        --name shoes \
        --display_port 30001 \
        --checkpoints_dir ./checkpoints/

Step 2 training command:

    python train.py \
        --netG resnet_12blocks \
        --loss_style_weight 2 \
        --loss_content_weight 1 \
        --pcploss_weight 0.05 \
        --L1loss_weight 10.0 \
        --display_freq 40 \
        --print_freq 100 \
        --load_size 140 \
        --crop_size 128 \
        --niter 100 \
        --niter_decay 600 \
        --input_nc 3 \
        --output_nc 3 \
        --dataroot ../datasets/shoes_step2 \
        --name shoes \
        --lambda_B 10.0 \
        --display_port 30000 \
        --batch_size 1 \
        --checkpoints_dir ./checkpoints/

