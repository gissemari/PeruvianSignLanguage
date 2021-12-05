Guide to make this model work
According to the original model ChaLearn, we are asked to create folders:
0. mkdir -p project/data/{mp4,kp,kpflow2} for Linux

1. Run "sh 1-erraseDataset.sh"

2. Create or activate environment. Use the file enviroment.yml

3. Run "sh 2-generateDataset.sh NUM_CLASSES" (replace NUM_CLASSES with the number of clases needed)

4. Manually modify the line 27 of the file "module.py" in "models" folder
   NUM_CLASSES =      <-- the same value writed in step 2

5. Run "sh 3-runModel.sh"
