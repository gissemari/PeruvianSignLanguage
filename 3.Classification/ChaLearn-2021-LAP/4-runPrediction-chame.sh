#python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_26/checkpoints/epoch=41-step=83.ckpt' --submission_template data/test_nframes.csv --out predOutputs.csv  --num_workers 4 --sequence_length 16 --temporal_stride 2 --learning_rate 1e-6 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 2 --num_heads 2 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5

#version_42
#python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_42/checkpoints/epoch=106-step=213.ckpt' --submission_template data/test_labels.csv --out predOutputs42.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 20 --temporal_stride 1 --learning_rate 1e-4 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF  --num_classes 5

#version_43
#python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_43/checkpoints/epoch=102-step=205.ckpt' --submission_template data/test_labels.csv --out predOutputs43.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 20 --temporal_stride 1 --learning_rate 1e-5 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5

#version_44
#python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_44/checkpoints/epoch=171-step=343.ckpt' --submission_template data/test_labels.csv --out predOutputs44.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 20 --temporal_stride 1 --learning_rate 1e-6 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5

#version_45
#python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_45/checkpoints/epoch=179-step=359.ckpt' --submission_template data/test_labels.csv --out predOutputs45.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 15 --temporal_stride 1 --learning_rate 1e-4 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5

#version_46
#python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_46/checkpoints/epoch=109-step=219.ckpt' --submission_template data/test_labels.csv --out predOutputs46.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 15 --temporal_stride 1 --learning_rate 1e-5 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5

#version_47
#python predict.py --log_dir project/log --dataset handcrop_poseflow  --checkpoint 'project/log/VTN_HCPF/version_47/checkpoints/epoch=103-step=207.ckpt' --submission_template data/test_labels.csv --out predOutputs47.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 15 --temporal_stride 1 --learning_rate 1e-6 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5

#version_34
 python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_34/checkpoints/epoch=108-step=217.ckpt' --submission_template data/test_labels.csv --out predOutputs34.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 15 --temporal_stride 2 --learning_rate 1e-5 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5

#version_35
python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_35/checkpoints/epoch=187-step=375.ckpt' --submission_template data/test_labels.csv --out predOutputs35.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 15 --temporal_stride 2 --learning_rate 1e-6 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5
 
 #version_37
python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_37/checkpoints/epoch=137-step=275.ckpt' --submission_template data/test_labels.csv --out predOutputs37.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 10 --temporal_stride 2 --learning_rate 1e-5 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5

#version_38
python predict.py --log_dir project/log --dataset handcrop_poseflow --checkpoint 'project/log/VTN_HCPF/version_38/checkpoints/epoch=100-step=201.ckpt' --submission_template data/test_labels.csv --out predOutputs38.csv --subject pucpSubject.csv --num_workers 4 --sequence_length 10 --temporal_stride 2 --learning_rate 1e-6 --gradient_clip_val=1 --gpus 1 --cnn rn34 --num_layers 4 --num_heads 4 --batch_size 8 --accumulate_grad_batches 8 --data_dir project/data/mp4 --model VTN_HCPF --num_classes 5