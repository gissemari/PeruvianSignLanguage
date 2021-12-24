source activate chalearn
python /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/train.py --log_dir /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/project/log --dataset handcrop_poseflow --num_workers 4 --sequence_length 16 --temporal_stride 2 --learning_rate 1e-4 --gradient_clip_val=1 --gpus 0 --cnn rn34 --num_layers 4 --num_heads 8 --batch_size 4 --accumulate_grad_batches 8 --data_dir /home/bejaranog/signLanguage/PeruvianSignLanguage/3.Classification/ChaLearn-2021-LAP/project/data/mp4 --model VTN_HCPF

