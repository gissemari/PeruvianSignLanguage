# Standard library imports
import argparse
import os
import sys
import random

# Third party imports
import pandas as pd
pd.set_option("display.max_colwidth", 15) 

# Local imports
sys.path.append('../')
import utils.video as uv

parser = argparse.ArgumentParser(description='generate a csv list of words in different orders')
parser.add_argument('--dict_input', type=str,  required=True, help='relative path of scv output set of landmarks.')
parser.add_argument('--seed', type=int, default=42)
#parser.add_argument('--csv_name', type=str, required=True)

args = parser.parse_args()

random_seed = args.seed
random.seed(random_seed)

dict_input = os.path.normpath(args.dict_input)
df = uv.get_list_from_json_dataset(dict_input)

unique_labels = df['label'].unique()

random.shuffle(unique_labels)

randomized_df = pd.DataFrame({'label': unique_labels})

csv_name = os.path.splitext(os.path.basename(dict_input))[0] + f'_V{args.seed}.csv' #args.csv_name
csv_dirname = os.path.dirname(dict_input)
csv_output_path = os.sep.join([csv_dirname, csv_name])

randomized_df.to_csv(csv_output_path, index=False, header=False)

print(randomized_df)

