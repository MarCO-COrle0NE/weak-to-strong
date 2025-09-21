import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['CIFAR100','CIFAR10',"Places365"]
#ST = ["s14","b14","l14"]
STS = ['dinov2','vit','mamba','sbb','resnet-d']
ST = {'dinov2':["s14","b14","l14","g14"],
      'vit':['tiny','small','base','large'],
      'mamba':['femto','kobe','tiny','small','base'],
      'sbb':["wee","pwee","little","medium","betwixt"],
      'resnet-d':["18d","34d","50d","101d","152d","200d"]}
TC=["18_a3"]
SUBSET = [1]
SUBSET_ST= [1]
#ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
#cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] + ['env3_out_loss']#+['agreement','agreement_neg','agreement_pos']
#weak_df = pd.read_json('weak_subset/weak.json')

def select_data(file):
    df = pd.read_json(file,lines=True)
    return df
    
def collect_contrastive(seed,ft=False,hundred=False):
    df_list = []
    for dataset in DATASETS:
        for sts in STS:
            # for st in ST[sts]:
                base_path = f'st/{seed}/{sts}/{dataset}'
                base_file = base_path + '/non-project/contrastive_loss.jsonl'
                df = select_data(base_file)
                df = df.drop_duplicates(subset=['seed','size'],keep='last')
                df['st'] = sts
                df['dataset'] = dataset
                df_list.append(df)
    df = pd.concat(df_list,ignore_index=True)
    #df.to_json(f'st/{seed}/non-project/contrastive_loss.json',index=False)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ft',action='store_true')
    parser.add_argument('--hundred',action='store_true')
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = collect_contrastive(args.seed,args.ft,args.hundred)
    #file = output_dir+'/all_accs.json'
    # with open(file,'w') as f:
    #     json.dump(result,f)

    # Save the records as JSON
    records.to_json(args.output_dir+'/contrastive_loss.json',index=False)
    # file = output_dir+'/all_accs_table.json'
    # with open(file, 'w') as json_file:
    #     json.dump(records, json_file, indent=4)
    print('done')
    #pprint.pprint(result)
    
