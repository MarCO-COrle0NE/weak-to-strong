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
ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
#cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] + ['env3_out_loss']#+['agreement','agreement_neg','agreement_pos']

def select_data(file):
    with open(file, 'r') as f:
    # Read the file's contents
        content = f.read()
    return float(content[-19:-1])

def all_accs_table(job,ft=False,twenty=False):
    records = []
    count = 0
    for subset in SUBSET:
            for tc in TC:
                    for dataset in DATASETS:
                            count+=1
                            file = f'eval_subset_{job}_{count}.out'
                            data = select_data(file)
                            records.append(
                                    {
                                        'dataset':dataset,
                                        'tc':tc,
                                        'subset':subset,
                                        'acc':data,
                                    }
                                )
    return records 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--job', type=int)
    parser.add_argument('--ft',action='store_true')
    parser.add_argument('--twenty',action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = all_accs_table(args.job,ft=args.ft,twenty=args.twenty)
    #file = output_dir+'/all_accs.json'
    # with open(file,'w') as f:
    #     json.dump(result,f)

    # Save the records as JSON
    file = output_dir+'/weak.json'
    with open(file, 'w') as json_file:
        json.dump(records, json_file, indent=4)
    print('done')
    #pprint.pprint(result)
    
