import pandas as pd
from collections import Counter
import os

options = {"1":"AEC",
           "2":"PUCP_PSL_DGI156"}

chosen = []
alert = []

while(True):
    print("########################################")
    for num, opt in options.items():
        print(f'{num}) {opt}')
    print('\nafter select two options, write "ok" to continue the process')
    print()

    if len(chosen) > 0:
        print("you choose this data to mix:",chosen) 
        print()

    if len(alert) > 0:
        print(alert.pop())

    ans = input("Choose an option:")
    print()

    if ans == 'ok':
        if len(chosen) < 2:
            alert.append("ALERT! - Please select at least two options\n")
            continue
        break

    if not ans.isnumeric():
        alert.append("ALERT! - Choose a valid option\n")
        continue

    if not ans in options.keys():
        alert.append("ALERT! - choose a number in a range of options\n")
        continue

    if ans in chosen:
        chosen.remove(ans)
    else:
        chosen.append(ans)

chosenOpt = []

for num, opt in options.items():
    if num in chosen:
        chosenOpt.append(opt)

print(chosenOpt)
print()

basePAth = 'Data/'

wordDict = dict()

LabelKey = 0

paths = []
words = []
labels = []
names = []
timestepsLen = []


for opt in chosenOpt:

    dictPath = basePAth+opt+'/dict.json'

    gloss = pd.read_json(dictPath)

    for glossIndex in gloss:

        word = gloss[glossIndex]["gloss"].upper()
    
        for inst in  gloss[glossIndex]["instances"]:

            if not word in wordDict.keys():
                wordDict[word] = LabelKey
                LabelKey += 1

            paths.append(inst["keypoints_path"])
            words.append(word)
            labels.append(wordDict[word])
            names.append(inst["unique_name"])
            timestepsLen.append(inst["frame_end"])

print("Before errase classes with 1 or 2 instances")
print("paths",len(paths))
print("words",len(words))
print("labels",len(labels))
print("names",len(names))
print("timestepLen",len(timestepsLen))

counter = Counter(words)

finalPaths = []
finalWords = []
finalLabel = []
finalNames = []
finalTimestepsLen = []

instanceLimit = 35
LabelKey = 0

#bannedList = ["???"]
bannedList = ["???", "","YA","QUÉ","QUÉ?","BIEN","DOS","","AHÍ","LUEGO","YO","ÉL","TÚ"]
#bannedList2 = ["???", "","ESE","QUÉ","QUÉ?","BIEN","DOS",]
#whiteList = ["ESE","QUÉ","QUÉ?","BIEN","DOS"]
#whiteList = ["CAMINAR","CASA","COMER","CÓMO","CUÁNTO","ESE","HOMBRE","MAMÁ","MUJER","NO","PENSAR","PORCENTAJE","PROTEÍNA","SÍ","UNO"]

finalWordDict = dict()

toSeeDetails = ["YA","ESE"]
keypointsDetail = dict()

for path, word, label, name ,timestepLen in zip(paths, words, labels, names, timestepsLen):
    
    if counter[word] < instanceLimit:
        continue

    if word in toSeeDetails:
        if word not in keypointsDetail.keys():
            keypointsDetail[word] = [timestepLen]
        else:
            keypointsDetail[word].append(timestepLen)
    #print(word)
    if word in bannedList:
        continue
    
    if not word in finalWordDict.keys():
        finalWordDict[word] = LabelKey
        LabelKey += 1

    finalPaths.append(path)
    finalWords.append(word)
    finalLabel.append(finalWordDict[word])
    finalNames.append(name)
    finalTimestepsLen.append(timestepLen)
        
print("\nFinal")
print("paths",len(finalPaths))
print("words",len(finalWords))
print("labels",len(finalLabel))
print("names",len(finalNames))
print("timestepsLen", len(finalTimestepsLen))
print()

chosenOpt.sort()

print(chosenOpt)

dirPath = "./Data/merged/" + '-'.join(chosenOpt)
print(dirPath)
os.makedirs(dirPath, exist_ok=True)

df = pd.DataFrame.from_dict({
    "paths":  finalPaths,
    "words":  finalWords,
    "labels": finalLabel,
    "names":  finalNames,
    "timestepsLen": finalTimestepsLen,
}, orient='index')

df_meaning = pd.DataFrame.from_dict(finalWordDict, orient='index')

df_meaning.to_json(dirPath+'/meaning.json')
df.to_json(dirPath+"/merged.json")
df.to_pickle(dirPath+"/merged.pkl")

merged = pd.read_pickle(dirPath+"/merged.pkl")
print(merged.T)

print(*list(df_meaning.T.columns))
print(len(df_meaning.T.columns))

hist = Counter(list(merged.T["words"]))
#print(hist)

print("\n to see some details in")
for key, val in keypointsDetail.items():
    print(key,val, sum(val)/len(val))