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

# Title
parser = argparse.ArgumentParser(description='To split the dataset metadata to use it posteriorly to feed an IA')

parser.add_argument('--dict_output', type=str,  required=True, help='relative path of scv output set of landmarks.')
parser.add_argument('--json_name', type=str,  required=True,)

parser.add_argument('--top', type=int, default=-1)
parser.add_argument('--random', type=int, default=-1)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--min_instances', type=int, default=-1)
parser.add_argument('--max_instances', type=int, default=-1)
parser.add_argument('--banned_words', action='store_true',)
parser.add_argument('--selected_words', action='store_true',)

args = parser.parse_args()


def limit_instances(group, max_instances_numb):
    if len(group) > max_instances_numb:
        return group.head(max_instances_numb)
    else:
        return group


dict_output = os.path.normpath(os.sep.join([args.dict_output,"dict.json"]))

filtered_df = uv.get_list_from_json_dataset(dict_output)



# Filter: Get the top # most frecuent labels
if args.top > 0:
    label_counts = filtered_df['label'].value_counts()
    top_labels = label_counts.head(args.top).index.tolist()
    filtered_df = filtered_df[filtered_df['label'].isin(top_labels)]
# Filter: Get all the instances of a randomly # labels 
elif args.random > 0:
    unique_labels = filtered_df['label'].unique()
    random.seed(args.random_seed)
    random_labels = random.sample(list(unique_labels), 100)
    filtered_df = filtered_df[filtered_df['label'].isin(random_labels)]

# Filter: Keep labels with more than # instances
if args.min_instances >= 0:
    filtered_df = filtered_df.groupby('label').filter(lambda x: len(x) > args.min_instances)

# Filter: Limit to a maximum of # instances per label
if args.max_instances > 3:
    filtered_df = filtered_df.groupby('label').apply(limit_instances, args.max_instances)

# Filter: List of labels to keep
if args.selected_words:
    labels_to_keep = pd.read_csv('selected_words.csv', header=None)
    filtered_df = filtered_df[filtered_df['label'].isin(labels_to_keep[0])]

# Filter:List of labels to remove
if args.banned_words:
    labels_to_remove = pd.read_csv('banned_words.csv', header=None)
    filtered_df = filtered_df[~filtered_df['label'].isin(labels_to_remove[0])]

# To reset the DataFrame to its initial form
filtered_df = filtered_df.reset_index(drop=True)

print(filtered_df)
print("Remember: The gloss_id value is changed when the dict.json is reconstructed")

uv.reconstruct_json(filtered_df, os.sep.join([args.dict_output, args.json_name]))