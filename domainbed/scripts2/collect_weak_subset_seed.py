import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['ColoredMNISTID']
#TC=["50"]
#SUBSET = [0.02,0.05,0.07,0.1,0.2,0.5,0.75,1]
#ST = ["s14","b14","l14"]
ST = ["resnet50"]
TC=["resnet18"]
# SUBSET = [0.02,0.05,0.07,0.1,0.2,0.5,0.75,1]
SUBSET = [75,100,140,280,420,650]
#ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
cols_interest = [f'env{i}_in_acc' for i in range(2)]+[f'env{i}_out_acc' for i in range(2)]+['epoch','loss','step'] + ['env1_out_loss','cluster_err']#+['agreement','agreement_neg','agreement_pos']

def select_data(file,step=None):
    df = pd.read_json(file,lines=True)
    df = df[cols_interest]
    # Find the index of the last occurrence of step=0
    #last_step_0_index = df[df['step'] == 0].index[-1]
    # Slice the DataFrame starting from the last step=0 row
    #df = df.iloc[last_step_0_index:]
    # Select models -- last
    if step:
        model_last = df[df.step==step].iloc[-1]
    else:
        model_last = df.iloc[-1]
    # Select models -- ground truth
    model_acc = df.loc[df['env1_in_acc'].idxmax()]
    # Select models -- source validation set
    model_loss = df.loc[df['env1_out_loss'].idxmin()]
    return {'last':model_last,'acc':model_acc,'loss':model_loss}

def all_accs_table(seed,step=None):
    records = []
    count = 0
    for subset in SUBSET:
            for tc in TC:
                    for dataset in DATASETS:
                            count+=1
                            path = f'{dataset}/tc/{seed}/{subset}_{tc}'
                            # path = f'{dataset}/tc/{subset}_{tc}'
                            file = path + '/results.jsonl'
                            # data = select_data(file) if dataset=='CIFAR100' else select_data(file,step)
                            data = select_data(file)
                            for trait,values in data.items():
                                records.append(
                                    {
                                        'dataset':dataset,
                                        'tc':tc,
                                        'model_selection':trait,
                                        'subset':subset,
                                        'val_acc':values.env0_out_acc,
                                        'in_acc':values.env1_in_acc,
                                        'test_acc':values.env1_out_acc,
                                        'loss':values.loss,
                                        'test_loss':values.env1_out_loss,
                                        'step':int(values.step),
                                        'epoch':int(values.epoch),
                                        'cluster_err':values.cluster_err
                                    }
                                )
    return records 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--seed', type=int)
    parser.add_argument('--step', type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    if args.step:
        records = all_accs_table(args.seed,args.step)
    else:
        records = all_accs_table(args.seed)
    #file = output_dir+'/all_accs.json'
    # with open(file,'w') as f:
    #     json.dump(result,f)

    # Save the records as JSON
    file = output_dir+'/weak.json'
    with open(file, 'w') as json_file:
        json.dump(records, json_file, indent=4)
    print('done')
    #pprint.pprint(result)
    
