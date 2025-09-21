import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['VLCS','PACS','OfficeHome']
ST = ["s14", "b14", "l14"]
TC=["s"]
# ALGORITHM = ['WeightedOutput','Average','WeightedVoting','MajorityVoting']
ALGORITHM = ['WeightedOutput']
weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] #+['agreement','agreement_neg','agreement_pos']

def select_data(file):
    df = pd.read_json(file,lines=True)
    df = df[cols_interest]
    df['source_acc'] = df[['env0_out_acc', 'env1_out_acc', 'env2_out_acc']].mean(axis=1)
    # Select models -- last
    model_last = df.iloc[-1]
    model_last['select'] = 'last'
    # Select models -- ground truth
    model_ground_truth = df.loc[df['env3_out_acc'].idxmax()]
    model_ground_truth['select'] = 'ground_truth'
    # Select models -- source validation set
    model_source_val = df.loc[df['source_acc'].idxmax()]
    model_source_val['select'] = 'source_val'
    #return {'last':model_last.env3_out_acc,'ground_truth':model_ground_truth.env3_out_acc,'source_val':model_source_val.env3_out_acc}
    return pd.concat([model_last,model_ground_truth,model_source_val], axis=1).T.reset_index(drop=True)

# def all_accs(ft=False, twenty=False):
#     result = {'acc':{},'pgr':{}}
#     for dataset in DATASETS:
#         result['acc'][dataset] = {}
#         result['pgr'][dataset] = {}
#         for st in ST:
#             result['acc'][dataset][st] = {}
#             result['pgr'][dataset][st] = {}
#             if 'b' in st:
#                 base_path = f'{dataset}/dinov'
#             else:
#                 base_path = f'{dataset}/dinov_{st[0]}'
#             #print(base_path)
#             base_file = base_path + '/results.jsonl'
#             ceiling = select_data(base_file)['ground_truth']
#             result['acc'][dataset][st]['ceiling'] = ceiling
#             result['pgr'][dataset][st]['ceiling'] = ceiling
#             for tc in TC:
#                 result['acc'][dataset][st][tc] = {}
#                 result['pgr'][dataset][st][tc] = {}
#                 for algo in ALGORITHM:
#                     if 'r' in algo:
#                         weak = weak_performances_avg[dataset][TC.index(tc)]
#                     else:
#                         weak = weak_performances[dataset][TC.index(tc)]
#                     if not ft:
#                         if twenty:
#                             path = f'{dataset}/st_dinov/{algo}_20_{tc}_{st}'
#                         else:
#                             path = f'{dataset}/st_dinov/{dataset}_{algo}_{tc}_{st}'
#                     else:
#                         path = f'{dataset}/st_dinov/{algo}_ft_{tc}_{st}'
#                     #print(path)
#                     file = path + '/results.jsonl'
#                     data = select_data(file)
#                     pgr = {k:(v-weak)/(ceiling-weak) for k,v in data.items()}
#                     result['acc'][dataset][st][tc][algo] = data
#                     result['pgr'][dataset][st][tc][algo] = pgr
#                     result['acc'][dataset][st][tc][algo]['weak_performance'] = weak
#                     result['pgr'][dataset][st][tc][algo]['weak_performance'] = weak
#     return result               
                
def all_accs_table(ft=False,twenty=False):
    records = []
    for dataset in DATASETS:
        for st in ST:
            if 'b' in st:
                base_path = f'{dataset}/dinov'
            else:
                base_path = f'{dataset}/dinov_{st[0]}'
            #print(base_path)
            base_file = base_path + '/results.jsonl'
            ceiling = select_data(base_file)
            ceiling['ft'] = False
            ceiling['dataset'] = dataset
            ceiling['st'] = st
            records.append(ceiling)

            path = f'{dataset}/ft_{st[0]}_s_lp'
                    #print(path)
            file = path + '/results.jsonl'
            data = select_data(file)
            data['ft'] = True
            data['dataset'] = dataset
            data['st'] = st
            records.append(data)
    return pd.concat(records,ignore_index=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--ft',action='store_true')
    parser.add_argument('--twenty',action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    records = all_accs_table(ft=args.ft,twenty=args.twenty)
    file = output_dir+'/ftlp.csv'
    records.to_csv(file,index=False)
    print(records)
    
