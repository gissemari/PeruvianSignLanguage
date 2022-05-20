import pandas as pd

basePAth = 'Data/'
opt = "PUCP_PSL_DGI156"
dictPath = basePAth+opt+'/dict.json'

gloss = pd.read_json(dictPath)

LSP = []

for glossIndex in gloss:
    #print(glossIndex)
    glossInstList = []
    for instPos in range(len(gloss[glossIndex]["instances"])):

        tmp = gloss[glossIndex]["instances"][instPos]["keypoints_path"]

        tmp = tmp.replace("//","/").split("/")
        tmp.insert(4,'pkl')
        tmp = '/'.join(tmp)

        glossInst = {
            "image_dimention": {
                "height": gloss[glossIndex]["instances"][instPos]["image_dimention"]["height"],
                "witdh": gloss[glossIndex]["instances"][instPos]["image_dimention"]["witdh"]
            },
            "keypoints_path": tmp,
            #"image_path": pklImagePath,
            "frame_end": gloss[glossIndex]["instances"][instPos]["frame_end"],
            "frame_start": gloss[glossIndex]["instances"][instPos]["frame_start"],
            "instance_id": gloss[glossIndex]["instances"][instPos]["instance_id"],
            "signer_id": gloss[glossIndex]["instances"][instPos]["signer_id"],
            "unique_name": gloss[glossIndex]["instances"][instPos]["unique_name"],
            "source": opt,
            "split": "",
            "variation_id": gloss[glossIndex]["instances"][instPos]["variation_id"],
            "source_video_name": gloss[glossIndex]["instances"][instPos]["source_video_name"],
            "timestep_vide_name": gloss[glossIndex]["instances"][instPos]["timestep_vide_name"]
        }

        glossInstList.append(glossInst)

    glossDict = {"gloss": gloss[glossIndex]["gloss"],
                "instances": glossInstList
                }
    LSP.append(glossDict)
df = pd.DataFrame(LSP)
df.to_json(dictPath, orient='index', indent=2)
