import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['ColoredMNISTID']
#ST = ["s14","b14","l14"]
ST = ["50"]
TC=["18"]
SEED=[0,1,2]
SUBSET = [75, 100, 140, 280, 420, 650]
#SUBSET_ST = [2800, 3220, 3640, 4116]
SUBSET_ST = [4550, 6370, 6825, 7280, 8190, 9100]
TC=["vit_tiny", "vit_tiny", "vit_tiny", "vit_tiny", "vit_tiny", "vit_tiny", "vit_base", "vit_base", "vit_base", "vit_base", "vit_base", "resnet18", "resnet18", "resnet18", "resnet18", "resnet18"]
SUBSET=[30,70,140,200,400,600,70,200,400,600,1000,200,600,1000,1400,1800]

ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
#cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] + ['env3_out_loss']#+['agreement','agreement_neg','agreement_pos']

def select_data(file):
    with open(file, 'r') as f:
    # Read the file's contents
        # content = f.read()
        last_line = f.readlines()[-1]
    result = last_line[:-1].split()[-1]
    return float(result)
    #return float(content[-19:-1])

def all_accs_table(job,end):
    records = []
    count = 0
    for i,subset in enumerate(SUBSET):
        tc=TC[i]
        for seed in SEED:
        # for subset_st in SUBSET_ST:
        #     for tc in TC:
        #         for st in ST:
                        for dataset in DATASETS:
                            count+=1
                            file = f'eval_mse_tc_{job}_{count}.out'
                            mse = select_data(file)
                            records.append(
                                    {
                                        'dataset':dataset,
                                        'tc':tc,
                                        'subset':subset,
                                        'seed': seed,
                                        'mse':mse,
                                    }
                                )
    return records 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--job', type=int)
    parser.add_argument('--end', type=int)
    parser.add_argument('--ft',action='store_true')
    parser.add_argument('--twenty',action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = all_accs_table(args.job,args.end)
    #file = output_dir+'/all_accs.json'
    # with open(file,'w') as f:
    #     json.dump(result,f)

    # Save the records as JSON
    file = output_dir+'/mse.json'
    with open(file, 'w') as json_file:
        json.dump(records, json_file, indent=4)
    print('done')
    #pprint.pprint(result)
    
