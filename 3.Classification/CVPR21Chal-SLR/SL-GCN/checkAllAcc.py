import os
import pandas as pd
from os.path import exists
from shutil import copyfile

file_name = "20-balanced-7classes.csv"

os.makedirs(f'final_models/{file_name[:-4]}', exist_ok = True)


base_path = 'work_dir/'
model_path = 'save_models/'
directories = os.listdir(base_path)

print("\nSummary:\n")

dict = {}

checked = {"joint":False,
           "bone":False,
           "joint_motion":False,
           "bone_motion":False}

for folder in directories:
    key = ""
    if folder.find('joint') != -1:
        key = key + 'joint'
    elif folder.find('bone') != -1:
        key = key + 'bone'
    if folder.find('motion') != -1:
        key = key + '_motion'

    path = base_path + folder + '/eval_results'
    files = os.listdir(path)

    #print(folder+':')

    best = -1
    num = -1
    finalModel = ''

    for file in files:
        if file == 'best_acc.pkl':
            continue

        pos = float(file[:-4].split('_')[1])
        acc = float(file[:-4].split('_')[2])


        if best < acc:
            num = int(pos)
            best = acc
            finalModel = file

    if not checked[key]:
        dict[key]={}
        checked[key]=True


    if best != -1:
        #print(f'No checkpoint in {path}')
        if folder.find("test") != -1:
            #print("Best acc -> ",best*100,f'({num})\n')
            dict[key].update({"TestAcc":best*100})
            #dict[key]={"TestAcc":best*100}
        else:
            #print("Best acc -> ",best*100,f'({num})')
            dict[key].update({"bestValAcc":best*100})
            dict[key].update({"bestValAccPos":pos-1})
            copyfile(model_path + f'sign_{key}_final-{num}.pt',f'final_models/{file_name[:-4]}/sign_{key}_final-{num}.pt')

    #print(dict)

    top1 = -1
    top5 = -1
    pos = -1

    #tempTop1 = -1

    f = open(base_path+folder+'/log.txt','r')

    for line in f:

        if line.find('Training epoch:') != -1:
            tempPos = int(line.split(':')[-1])
        if line.find('Top1:')!= -1:
            tempTop1 = float(line.split(':')[-1][:-2])
        elif line.find('Top5:')!= -1:
            tempTop5 = float(line.split(':')[-1][:-2])
            if tempTop1 >= top1:
                top1 = tempTop1
                top5 = tempTop5
                pos = tempPos
    if top1 != -1:
        trainNum = int(pos-1)
        dict[key].update({"EpocPos":int(pos-1)})
        dict[key].update({"ValTop1":top1})
        dict[key].update({"ValTop5":top5})
        copyfile(model_path + f'sign_{key}_final-{trainNum}.pt',f'final_models/{file_name[:-4]}/sign_{key}_final-{trainNum}.pt')

        #print("epoch:",pos-1)
        #print("top1",top1)
        #print("top5",top5)
        #print()

    f.close()

df = pd.DataFrame(dict)
df = df.T
print(df)
print()

if exists(file_name):
    question = input("Do you want to replace the file "+file_name+"? (y/n): ")
    if question == ("y"):
        df.to_csv(file_name)
        print("yes")
    else:
        print("no")
else:
    df.to_csv(file_name)
