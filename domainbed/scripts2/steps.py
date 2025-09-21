import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['CIFAR100','CIFAR10']
#ST = ["s14","b14","l14"]
ST = ["b14"]
TC=["50"]
SUBSET = [0.02,0.05,0.07,0.1,0.2,0.5,0.75,1]
SUBSET_ST= [0.05,0.2,0.5,1]
#ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
#cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] + ['env3_out_loss']#+['agreement','agreement_neg','agreement_pos']
#weak_df = pd.read_json('weak_subset/weak.json')
cols_interest = [f'env{i}_in_acc' for i in range(2)]+[f'env{i}_out_acc' for i in range(3)]+['epoch','loss','step'] + ['env2_out_loss']+['agreement','agreement_neg','agreement_pos','cluster_err','cluster_err_all']

def time_data(file,all=True):
    df = pd.read_json(file,lines=True)
    # Find the index of the last occurrence of step=0
    last_step_0_index = df[df['step'] == 0].index[-1]
    # Slice the DataFrame starting from the last step=0 row
    df = df.iloc[last_step_0_index:]
    df = df[cols_interest]
    return df
    
def all_accs_table(seed):
    #weak_df = pd.read_json(f'weak/{seed}/{TC[0]}/weak.json')
    records = []
    for dataset in DATASETS:
        for st in ST:
            # base_path = f'{dataset}/dinov_{st[0]}'
            # #print(base_path)
            # base_file = base_path + '/results.jsonl'
            # ceiling = select_data(base_file)['ground_truth'].env3_out_acc
            for tc in TC:
                    for subset in SUBSET:
                        for subset_st in SUBSET_ST:
                            #weak = weak_df[(weak_df.dataset==dataset)&(weak_df.subset==subset)&(weak_df.tc==int(tc))&(weak_df.model_selection=='last')].test_acc.item()
                            path = f'{dataset}/st/{seed}/{subset}_{tc}_{subset_st}_{st}'
                            #print(path)
                            file = path + '/results.jsonl'
                            data = time_data(file)
                            data['tc'] = tc
                            data['st'] = st
                            data['dataset'] = dataset
                            data['subset'] = subset
                            data['subset_st'] = subset_st
                            data['seed'] = seed
                            records.append(data)
    records_df = pd.concat(records, ignore_index=True)
    return records_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ft',action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = all_accs_table(args.seed)
    file = output_dir+'/steps.csv'
    records.to_csv(file,index=False)
    print('done')
    #pprint.pprint(result)
    
