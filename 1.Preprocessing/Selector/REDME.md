This script is used after finish all the process to create Dict.json

It takes the desired words and using Dict.json create:

- X.data
- Y.data
- y_label.data
- weight.data
- X_timeSteps.data

There are a few options you have for creating those data files (hyperparameters):

- shuffle: to have it with random order
- leastValue: find the class with the least number of instance and take that number of instances for all the classes
