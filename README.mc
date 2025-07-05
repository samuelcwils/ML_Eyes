To begin using this library...

train.py is the file where the model is trained, it pulls in information from Kvasir-SEG and pulls in
what the model should look like from other files.

Some example arguments

With optuna: python train.py --seed 500 --model_type unet_2d --epochs 5 --n_trials 3 --batch_size 4 --optimizer adam --loss dice --im_width 256 --im_height 256 --learning_rate 0.001 0.1 --n_filters 64 128 --layer_depth 5 5 --stack_num_down 2 2 --stack_num_up 1 2 --activation GELU --output_activation Sigmoid --use_neptune --use_optuna

Without optuna: python train.py --seed 500 --model_type unet_2d --epochs 5 --batch_size 4 --optimizer adam --loss dice --im_width 256 --im_height 256 --learning_rate 0.001 --n_filters 64 --layer_depth 5 --stack_num_down 2 --stack_num_up 1 --activation GELU --output_activation Sigmoid
