 python train.py --log_dir project/log --dataset handcrop_poseflow --num_workers 4 --sequence_length 16 --temporal_stride 2 --learning_rate 1e-4 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 8 --batch_size 4 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF

