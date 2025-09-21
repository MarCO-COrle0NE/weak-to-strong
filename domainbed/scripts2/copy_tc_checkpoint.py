import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint
import shutil

DATASETS = ['VLCS','PACS','OfficeHome']
TC=["50"]
SUBSET = [0.02,0.05,0.07,0.1,0.2,0.5,0.75,1]
#ALGORITHM = ['WeightedOutput','Average','WeightedVoting','MajorityVoting']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] #+['agreement','agreement_neg','agreement_pos']

def copy_checkpoint(path,env):
    file = path+'/results.jsonl'
    df = pd.read_json(file,lines=True)
    df = df[cols_interest]
    # Find the index of the last occurrence of step=0
    last_step_0_index = df[df['step'] == 0].index[-1]
    # Slice the DataFrame starting from the last step=0 row
    df = df.iloc[last_step_0_index:]
    col = f'env{env}_out_acc'
    # Select models -- ground truth
    model_ground_truth = df.loc[df[col].idxmax()]
    step = model_ground_truth.step
    print(env,step)
    checkpoint = path+'/model_step'+str(int(step))+'.pkl'
    dsc = path+'/model_ground_truth.pkl'
    shutil.copy(checkpoint,dsc)
    #return {'last':model_last[col],'last_eval':model_last.env3_out_acc,'ground_truth':model_ground_truth[col],'ground_truth_eval':model_ground_truth.env3_out_acc}
    
def copy_not_used(path,env):
    checkpoint = path+'/0/model.pkl'
    dsc = path+'/3/model_ground_truth.pkl'
    shutil.copy(checkpoint,dsc)
                
def all_accs_table(seed):
    for dataset in DATASETS:
        print(dataset)
        for subset in SUBSET:
            print(subset)
            for tc in TC:
                for env in [0,1,2]:
                    if seed == 0:
                        path = f'{dataset}/tc_subset/{subset}_{tc}/{env}'
                    else:
                        path = f'{dataset}/tc_subset/{seed}/{subset}_{tc}/{env}'
                    #print(path)
                    copy_checkpoint(path,env)
                env = 3
                if seed == 0:
                        path = f'{dataset}/tc_subset/{subset}_{tc}/{env}'
                else:
                        path = f'{dataset}/tc_subset/{seed}/{subset}_{tc}/{env}'
                os.makedirs(path,exist_ok=True)
                if seed == 0:
                        path = f'{dataset}/tc_subset/{subset}_{tc}'
                else:
                        path = f'{dataset}/tc_subset/{seed}/{subset}_{tc}'
                copy_not_used(path,env)
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = all_accs_table(args.seed)
    print('done')
    #pprint.pprint(result)
    
