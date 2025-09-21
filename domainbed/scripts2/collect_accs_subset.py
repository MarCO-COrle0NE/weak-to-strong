import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['VLCS','PACS','OfficeHome']
#ST = ["s14","b14","l14"]
ST = ["b14"]
TC=["50"]
SUBSET = [0.02,0.05,0.07,0.1,0.2,0.5,0.75,1]
SUBSET_ST= [0.05,0.2,0.5,1]
ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
#cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] + ['env3_out_loss']#+['agreement','agreement_neg','agreement_pos']
#weak_df = pd.read_json('weak_subset/weak.json')

def select_data(file,file2=None):
    df = pd.read_json(file,lines=True)
    #df = df[cols_interest]
    if file2:
        df2 = pd.read_json(file2,lines=True)
        df = pd.concat([df,df2],ignore_index=True)
    df['source_acc'] = df[['env0_out_acc', 'env1_out_acc', 'env2_out_acc']].mean(axis=1)
    # Select models -- last
    model_last = df.iloc[-1]
    # Select models -- ground truth
    model_ground_truth = df.loc[df['env3_out_acc'].idxmax()]
    # Select models -- source validation set
    model_source_val = df.loc[df['source_acc'].idxmax()]
    return {'last':model_last,'ground_truth':model_ground_truth,'source_val':model_source_val}
    
def all_accs_table(seed,ft=False,twenty=False):
    if seed == 0:
        weak_df = pd.read_json('weak_subset/weak.json')
    else:
        weak_df = pd.read_json(f'weak_subset/{seed}/weak.json')
    records = []
    for dataset in DATASETS:
        for st in ST:
            if 'b' in st:
                base_path = f'{dataset}/dinov'
            else:
                base_path = f'{dataset}/dinov_{st[0]}'
            #print(base_path)
            base_file = base_path + '/results.jsonl'
            ceiling = select_data(base_file)['ground_truth'].env3_out_acc
            for tc in TC:
                for algo in ALGORITHM:
                    for subset in SUBSET:
                        for subset_st in SUBSET_ST:
                            weak = weak_df[(weak_df.dataset==dataset)&(weak_df.subset==subset)&(weak_df.tc==int(tc))].acc.item()
                            if seed == 0:
                                path = f'{dataset}/st_dinov_subset/{algo}_{subset}_{tc}_{subset_st}_{st}'
                            else:
                                path = f'{dataset}/st_dinov_subset/{seed}/{algo}_{subset}_{tc}_{subset_st}_{st}'
                            #print(path)
                            file = path + '/results.jsonl'
                            data = select_data(file)
                            for trait,values in data.items():
                                acc = values.env3_out_acc
                                records.append(
                                    {
                                        'dataset':dataset,
                                        'tc':tc,
                                        'st':st,
                                        'algo':algo,
                                        'model_selection':trait,
                                        'weak':weak,
                                        'subset':subset,
                                        'subset_st':subset_st,
                                        'ceiling':ceiling,
                                        'acc':acc,
                                        'in_acc':values.env3_in_acc,
                                        'loss':values.env3_out_loss,
                                        'agreement':values.agreement,
                                        'agreement_neg':values.agreement_neg,
                                        'agreement_pos':values.agreement_pos,
                                        'step':int(values.step),
                                        'epoch':int(values.epoch),
                                        'pgr':(acc-weak)/(ceiling-weak)
                                    }
                                )
    return records 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ft',action='store_true')
    parser.add_argument('--twenty',action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = all_accs_table(args.seed,ft=args.ft,twenty=args.twenty)
    #file = output_dir+'/all_accs.json'
    # with open(file,'w') as f:
    #     json.dump(result,f)

    # Save the records as JSON
    file = output_dir+'/all_accs_table.json'
    with open(file, 'w') as json_file:
        json.dump(records, json_file, indent=4)
    print('done')
    #pprint.pprint(result)
    
