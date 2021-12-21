Guide to make this model work

0. According to the original model ChaLearn, we are asked to create folders. If in Windows, create manually:
	mkdir -p project/data/{mp4,kp,kpflow2} 
	mkdir -p project/data/mp4/{train,val,test}

1. Run "sh 1-erraseDataset.sh"

2. Create or activate environment. Use the file enviroment.yml

3. This step takes the dataset from the folder Data and preprocess it to have it in the ChaLearn structure (train_ids, labels and nframes)
Run "sh 2-generateDataset.sh NUM_CLASSES" (replace NUM_CLASSES with the number of clases needed). Here we expect to create the [name_video]_color.mp4 and the [name_video]_nframes in the mp4/train and mp4/val folders

4. Manually modify the line 27 of the file "module.py" in "models" folder
   NUM_CLASSES =      <-- the same value writed in step 2

5. Modify depending on the use of GPU or CPU the parameter on 3-runModel.sh --gpus
6. Run "sh 3-runModel.sh"
