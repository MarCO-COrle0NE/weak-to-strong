import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['VLCS','PACS','OfficeHome']
TC=["50"]
SUBSET = [0.02,0.05,0.07,0.1,0.2,0.5,0.75,1]
#ALGORITHM = ['WeightedOutput','Average','WeightedVoting','MajorityVoting']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] #+['agreement','agreement_neg','agreement_pos']

def select_data(file,env):
    df = pd.read_json(file,lines=True)
    df = df[cols_interest]
    col = f'env{env}_out_acc'
    # Select models -- last
    model_last = df.iloc[-1]
    # Select models -- ground truth
    model_ground_truth = df.loc[df[col].idxmax()]
    print(env,model_ground_truth.step)
    return {'last':model_last[col],'last_eval':model_last.env3_out_acc,'ground_truth':model_ground_truth[col],'ground_truth_eval':model_ground_truth.env3_out_acc}
    
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
        print(dataset)
        for subset in SUBSET:
            print(subset)
            for tc in TC:
                for env in [0,1,2]:
                    path = f'{dataset}/tc_subset/{subset}_{tc}/{env}'
                    #print(path)
                    file = path + '/results.jsonl'
                    data = select_data(file,env)
                    for trait in ['ground_truth','last']:
                        records.append(
                            {
                                'dataset':dataset,
                                'model':tc,
                                'subset':subset,
                                'env':env,
                                'model_selection':trait,
                                'eval':data[f'{trait}_eval'],
                                'acc':data[trait],
                            }
                        )
    return records 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = all_accs_table()
    file = output_dir+'/all_accs.json'
    # with open(file,'w') as f:
    #     json.dump(result,f)

    # Save the records as JSON
    file = output_dir+'/all_accs_table.json'
    with open(file, 'w') as json_file:
        json.dump(records, json_file, indent=4)
    print('done')
    #pprint.pprint(result)
    
