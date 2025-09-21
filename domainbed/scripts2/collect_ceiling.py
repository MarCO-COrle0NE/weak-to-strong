import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['CIFAR100','CIFAR10',"Places365"]
#TC=["50"]
#SUBSET = [0.02,0.05,0.07,0.1,0.2,0.5,0.75,1]
#ST = ["s14","b14","l14"]
#ST = ["b14"]
ST = {'dinov2':["s14","b14","l14","g14"],
      'vit':['tiny','small','base','large'],
      'mamba':['femto','kobe','tiny','small','base'],
      'sbb':["wee","pwee","little","medium","betwixt"],
      'resnet-d':["18d","34d","50d","101d","152d","200d"]}
TC=["18_a3"]
#SUBSET= [0.05,0.2,0.5,1]
SUBSET=[1]
SEEDS=[0,1,2,3,4]

recorded_res = {'CIFAR100':{'200d':[60.26,60.27,60.29,60.35]}, # for seed 1,2,3,4
                'CIFAR10':{'152d':[85.1,85.1,85.06,85.11],'200d':[85.46,85.35,85.38,85.41]},
                "Places365":{"101d":[44.337,44.42191720,44.3863,44.40822],"152d":[45.40548,45.52876651,45.3643383,45.463],"200d":[43.8055,43.68767,43.7643826,43.84657433]}}
#ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
#cols_interest = [f'env{i}_in_acc' for i in range(2)]+[f'env{i}_out_acc' for i in range(3)]+['epoch','loss','step'] + ['env2_out_loss']#+['agreement','agreement_neg','agreement_pos']
cols_interest = ['env1_in_acc']+[f'env{i}_out_acc' for i in range(1,3)]+['epoch','loss','step'] + ['env2_out_loss']

def select_data(file):
    df = pd.read_json(file,lines=True)
    df = df[cols_interest]
    # Find the index of the last occurrence of step=0
    #last_step_0_index = df[df['step'] == 0].index[-1]
    # Slice the DataFrame starting from the last step=0 row
    #df = df.iloc[last_step_0_index:]

    # Select models -- last
    model_last = df.iloc[-1]
    # Select models -- ground truth
    model_ground_truth = df.loc[df['env2_out_acc'].idxmax()]
    # Select models -- source validation set
    model_source_val = df.loc[df['env1_out_acc'].idxmax()]
    return {'last':model_last,'gen':model_ground_truth,'val':model_source_val}

def all_accs_table(seed=None):
    records = []
    count = 0
    for subset in SUBSET:
        for seed in SEEDS:
            for st in ST:
                sts = st
                if sts == 'vit':
                    tail = '_vit'
                    ceiling_tail = ''
                elif sts == 'clip':
                    tail = '_clip'
                    ceiling_tail = tail
                elif sts == 'mamba':
                    tail = '_mamba'
                    ceiling_tail = tail
                elif sts == 'resnet-d':
                    tail = '_resnet-d'
                    ceiling_tail = tail
                elif sts == 'sbb':
                    tail = '_sbb'
                    ceiling_tail = tail
                else:
                    tail = ''
                    ceiling_tail = ''
                for size in ST[st]:
                    for dataset in DATASETS:
                            count+=1
                            
                            path = f'{dataset}/ceiling{ceiling_tail}/{seed}/{subset}_{size}'
                            file = path + '/results.jsonl'
                            try:
                                data = select_data(file)
                            except:
                                #if st == 'resnet-d' and seed != 0 and size in recorded_res[dataset]:
                                try:
                                    records.append(
                                        {
                                            'dataset':dataset,
                                            'st':st,
                                            'size':size,
                                            'model_selection':'gen',
                                            'subset_st':subset,
                                            'val_acc':0,
                                            'in_acc':0,
                                            'test_acc':recorded_res[dataset][size][seed-1],
                                            'loss':0,
                                            'test_loss':0,
                                            'step':0,
                                            'epoch':0,
                                            'seed':seed
                                        }
                                    )
                                    continue
                                except:
                                    path = f'{dataset}/ceiling{ceiling_tail}/0/{subset}_{size}'
                                    file = path + '/results.jsonl'
                                    data = select_data(file)
                            for trait,values in data.items():
                                records.append(
                                    {
                                        'dataset':dataset,
                                        'st':st,
                                        'size':size,
                                        'model_selection':trait,
                                        'subset_st':subset,
                                        'val_acc':values.env1_out_acc,
                                        'in_acc':values.env1_in_acc,
                                        'test_acc':values.env2_out_acc,
                                        'loss':values.loss,
                                        'test_loss':values.env2_out_loss,
                                        'step':int(values.step),
                                        'epoch':int(values.epoch),
                                        'seed':seed
                                    }
                                )
    return records 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = all_accs_table(args.seed)
    #file = output_dir+'/all_accs.json'
    # with open(file,'w') as f:
    #     json.dump(result,f)

    # Save the records as JSON
    file = output_dir+'/ceiling.json'
    with open(file, 'w') as json_file:
        json.dump(records, json_file, indent=4)
    print('done')
    #pprint.pprint(result)
    
