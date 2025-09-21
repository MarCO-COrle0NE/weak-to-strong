import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint
from pathlib import Path

DATASETS = ['ColoredMNISTID']
#ST = ["s14","b14","l14"]
ST = ["b14"]
TC=["resnet18",'mobile','alexnet','mamba']
SUBSET = [0.02,0.05,0.07,0.1,0.2,0.5,0.75,1]
SUBSET_ST= [0.02,0.05,0.1,0.2,0.5,1]
#ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
#cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] + ['env3_out_loss']#+['agreement','agreement_neg','agreement_pos']
#weak_df = pd.read_json('weak_subset/weak.json')

def select_data(file):
    df = pd.read_json(file,lines=True)
    # Find the index of the last occurrence of step=0
    # last_step_0_index = df[df['step'] == 0].index[-1]
    # # Slice the DataFrame starting from the last step=0 row
    # df = df.iloc[last_step_0_index:]
    # # Select models -- last
    # model_last = df.iloc[-1]
    # # Select models -- ground truth
    model_ground_truth = df.loc[df['env1_out_acc'].idxmax()]
    # # Select models -- source validation set
    # model_source_val = df.loc[df['env0_out_acc'].idxmax()]
    return model_ground_truth.env1_out_acc, model_ground_truth.env1_out_loss, model_ground_truth.step, model_ground_truth.cluster_err
    #return {'last':model_last,'gen':model_ground_truth,'val':model_source_val}
    
def all_accs_table(seed,ft=False,hundred=False):
    records = []
    for dataset in DATASETS:
        for tc in TC:
                    directory_path = Path(f'{dataset}/tc_intrinsic_did')
                    for folder in directory_path.iterdir():
                        if folder.is_dir() and tc in folder.name:
                            path = f'{dataset}/tc_intrinsic_did/{folder.name}'
                            #print(path)
                            dim = folder.name.split('_')[-1]
                            file = path + '/results.jsonl'
                            acc,loss,step,cluster_err = select_data(file)
                            # for trait,values in data.items():
                            records.append(
                                    {
                                        'dataset':dataset,
                                        'model':tc,
                                        'acc':acc,
                                        'test_loss':loss,
                                        'step':int(step),
                                        'cluster_err':cluster_err,
                                        'intrinsic_dim':int(dim)
                                    }
                                )
    return records 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--dataset', type=str, default="ColoredMNISTID")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ft',action='store_true')
    parser.add_argument('--hundred',action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = all_accs_table(args.seed,args.ft,args.hundred)
    #file = output_dir+'/all_accs.json'
    # with open(file,'w') as f:
    #     json.dump(result,f)

    # Save the records as JSON
    file = output_dir+'/all_accs_table.json'
    with open(file, 'w') as json_file:
        json.dump(records, json_file, indent=4)
    print('done')
    #pprint.pprint(result)
    
