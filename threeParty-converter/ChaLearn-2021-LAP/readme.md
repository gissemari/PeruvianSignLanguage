Guide to make this model work

1. Run "sh 1-erraseDataset.sh"

2. Run "sh 2-generateDataset.sh NUM_CLASSES" (replace NUM_CLASSES with the number of clases needed)

3. Manually modify the line 27 of the file "module.py" in "models" folder
   NUM_CLASSES =      <-- the same value writed in step 2

3. Run "sh 3-runModel.sh"
