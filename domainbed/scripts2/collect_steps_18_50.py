import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['VLCS','PACS','OfficeHome']
ST = ["18", "50"]
TC=["18", "50", "s"]
ALGORITHM = ['Average']
weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] #+['agreement','agreement_neg','agreement_pos']
cols_interest_all = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] +['agreement','agreement_neg','agreement_pos']

def time_data(file,all=True):
    df = pd.read_json(file,lines=True)
    if all:
        df = df[cols_interest_all]
    else:
        df = df[cols_interest]
    return df          
                
def all_accs_table(ft=False,twenty=False):
    records = []
    ceilings = []
    for dataset in DATASETS:
        for st in ST:
            base_path = f'{dataset}/base_{st}_lp'
            #print(base_path)
            base_file = base_path + '/results.jsonl'
            ceiling = time_data(base_file,all=False)
            ceiling['dataset'] = dataset
            ceiling['st'] = st
            ceilings.append(ceiling)
            for tc in TC:
                for algo in ALGORITHM:
                    if 'r' in algo:
                        weak = weak_performances_avg[dataset][TC.index(tc)]
                    else:
                        weak = weak_performances[dataset][TC.index(tc)]
                    if not ft:
                        path = f'{dataset}/st_dinov/{algo}_{tc}_{st}'
                    else:
                        path = f'{dataset}/st_dinov/{algo}_ft_{tc}_{st}'
                    #print(path)
                    file = path + '/results.jsonl'
                    data = time_data(file)
                    ceiling_acc = ceiling.env3_out_acc.max()
                    data['dataset'] = dataset
                    data['ceiling'] = ceiling_acc
                    data['weak'] = weak
                    data['tc'] = tc
                    data['st'] = st
                    data['algo'] = algo
                    data['pgr'] = (data.env3_out_acc-weak)/(ceiling_acc-weak)
                    records.append(data)
                    # for trait,acc in data.items():
                    #     records.append(
                    #         {
                    #             'dataset':dataset,
                    #             'tc':tc,
                    #             'st':st,
                    #             'algo':algo,
                    #             'model_selection':trait,
                    #             'weak':weak,
                    #             'ceiling':ceiling,
                    #             'acc':acc,
                    #             'pgr':(acc-weak)/(ceiling-weak)
                    #         }
                    #     )
    records_df = pd.concat(records, ignore_index=True)
    ceilings_df = pd.concat(ceilings, ignore_index=True)
    return records_df,ceilings_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--ft',action='store_true')
    parser.add_argument('--twenty',action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    records,ceilings = all_accs_table(ft=args.ft,twenty=args.twenty)
    file = output_dir+'/ceilings.csv'
    ceilings.to_csv(file,index=False)
    file = output_dir+'/records.csv'
    records.to_csv(file,index=False)
    print(records.head())
    print(ceilings.head())

    
