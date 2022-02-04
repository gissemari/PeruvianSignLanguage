
import sys

versionName = "joe"
csv_name = "new-summary.csv"
num_classes = 5

seq_lenList = [10]
strideList = [1, 2]
lrn_rtList = [1e-4, 1e-6]
resnetList = ["rn18"] #, "rn34"]
seedList = [11,31]


errorList = []

counter = 0

for seq_len in seq_lenList:
    for stride in strideList:
        for lrn_rt in lrn_rtList:
            for resnet in resnetList:
                for seed in seedList:
                    
                    counter = counter + 1
                    version = "%s_%d"%(versionName, counter)
                    checkpoint = "project/log/VTN_HCPF/%s/checkpoints/bestLoggedModel.ckpt"%(version)
                    out = "submission/predOutputs_%s.csv"%(version)
                    
                    a_script_descriptor = open("train.py")
                    a_script = a_script_descriptor.read()
                    
                    sys.argv = ["train.py",
                                "--log_dir", "project/log",
                                "--dataset", "handcrop_poseflow",
                                "--num_workers", "4",
                                "--sequence_length" , str(seq_len),
                                "--temporal_stride" , str(stride),
                                "--learning_rate" , str(lrn_rt),
                                "--gradient_clip_val=1",
                                "--gpus", "1",
                                "--cnn", resnet,
                                "--num_layers", "4",
                                "--num_heads", "8",
                                "--batch_size", "4",
                                "--accumulate_grad_batches", "8",
                                "--data_dir", "project/data/mp4",
                                "--model", "VTN_HCPF",
                                "--num_classes", str(num_classes),
                                "--seed", str(seed),
                                "--version", version,
                                "--csv_name", csv_name]
                    try:
                        exec(a_script)
                    except:
                        errorList.append({"seq_len":seq_len,
                                          "stride":stride,
                                          "lrn_rt":lrn_rt,
                                          "resnet":resnet})
                    a_script_descriptor.close()

                    b_script_descriptor = open("predict.py")
                    b_script = b_script_descriptor.read()

                    sys.argv = ["predict.py",
                                "--log_dir", "project/log",
                                "--dataset", "handcrop_poseflow",
                                "--num_workers", "4",
                                "--sequence_length" , str(seq_len),
                                "--temporal_stride" , str(stride),
                                "--learning_rate" , str(lrn_rt),
                                "--gradient_clip_val=1",
                                "--gpus", "1",
                                "--cnn", resnet,
                                "--num_layers", "4",
                                "--num_heads", "8",
                                "--batch_size", "4",
                                "--accumulate_grad_batches", "8",
                                "--data_dir", "project/data/mp4",
                                "--model", "VTN_HCPF",
                                "--num_classes", str(num_classes),
                                "--seed", str(seed),
                                "--version", version,
                                "--csv_name", csv_name,
                                "--checkpoint", checkpoint,
                                "--submission_template", "data/test_labels.csv",
                                "--out", out,
                                "--subject", "pucpSubject.csv"]
                    try:
                        exec(b_script)
                    except:
                        print("\n doesn't read bestLoggedModel.ckpt, trying with bestLoggedModel-v1.ckpt")
                    checkpoint = "project/log/VTN_HCPF/%s/checkpoints/bestLoggedModel-v1.ckpt"%(version)
                    sys.argv = ["predict.py",
                                "--log_dir", "project/log",
                                "--dataset", "handcrop_poseflow",
                                "--num_workers", "4",
                                "--sequence_length" , str(seq_len),
                                "--temporal_stride" , str(stride),
                                "--learning_rate" , str(lrn_rt),
                                "--gradient_clip_val=1",
                                "--gpus", "1",
                                "--cnn", resnet,
                                "--num_layers", "4",
                                "--num_heads", "8",
                                "--batch_size", "4",
                                "--accumulate_grad_batches", "8",
                                "--data_dir", "project/data/mp4",
                                "--model", "VTN_HCPF",
                                "--num_classes", str(num_classes),
                                "--seed", str(seed),
                                "--version", version,
                                "--csv_name", csv_name,
                                "--checkpoint", checkpoint,
                                "--submission_template", "data/test_labels.csv",
                                "--out", out,
                                "--subject", "pucpSubject.csv"]
                    try:
                        exec(b_script)
                    except:
                        print("doesn't read bestLoggedModel-v1.ckpt... if also bestLoggedModel.ckpt did not be readed, please run it manually")
                    
                    b_script_descriptor.close()

print("segmentation fault in:") 
print(errorList)
