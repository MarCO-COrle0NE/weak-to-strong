import pandas as pd
import json
import numpy as np
import argparse
import os
import sys
import pprint

DATASETS = ['CIFAR100','CIFAR10']
#ST = ["s14","b14","l14"]
#ST = ["tiny","small","large",  "base"]
#ST = {"vit":["tiny","small", "base", "large"],"clip":["base","large"],"dinov2":["s14","b14","l14","g14"],"mamba":["tiny","small","base","large"],"50":["50"],"dino":["b16"]}
ST = {'dinov2':["s14","b14","l14","g14"],
      'vit':['tiny','small','base','large'],
      'mamba':['femto','kobe','tiny','small','base'],
      'sbb':["wee","pwee","little","medium","betwixt"],
      'resnet-d':["18d","34d","50d","101d","152d","200d"]}
TC=["18_a3"]
#SUBSET = [0.02,0.05,0.07,0.1,0.2,0.5,0.75,1]
SUBSET = [1]
#SUBSET_ST= [0.02,0.05,0.1,0.2,0.5,1]
SUBSET_ST = [1]
#ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
#cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] + ['env3_out_loss']#+['agreement','agreement_neg','agreement_pos']
#weak_df = pd.read_json('weak_subset/weak.json')
ceiling_clip = {'CIFAR10':{'base':0.9654,'large':0.9682},
                'CIFAR100':{'base':0.8293,'large':0.8278}}
def select_data(file):
    df = pd.read_json(file,lines=True)
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

    
def all_accs_table(seed,ft=False,dataset="Places365",sts="dinov2"):
    # try:
    #     if dataset != 'Places365':
    #         weak_df = pd.read_json(f'weak/{seed}/{TC[0]}/weak.json')
    #     else:
    #         weak_df = pd.read_json(f'weak_Places/{seed}/{TC[0]}/weak.json')
    # except:
    #     if dataset != 'Places365':
    #         weak_df = pd.read_json(f'weak/0/{TC[0]}/weak.json')
    #     else:
    #         weak_df = pd.read_json(f'weak_Places/0/{TC[0]}/weak.json')
    weak_df = pd.read_json(f'weak/0/{TC[0]}/weak.json')
    records = []
    stl = ST[sts]
    # if sts != 'mamba' and dataset == 'Places365':
    #     stl = stl[:-1]
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
    if True:
        for st in stl:
            # if st == 's14' and dataset == 'Places365':
            #     continue
            for subset_st in SUBSET_ST:
                if sts == 'clip':
                    ceiling = ceiling_clip[dataset][st]
                else:
                    ceiling = f'ceiling{ceiling_tail}'
                    try:
                        base_path = f'{dataset}/{ceiling}/{seed}/{subset_st}_{st}'
                        base_file = base_path + '/results.jsonl'
                        ceiling = select_data(base_file)['gen'].env2_out_acc
                    except:
                        base_path = f'{dataset}/{ceiling}/0/{subset_st}_{st}'
                        base_file = base_path + '/results.jsonl'
                        ceiling = select_data(base_file)['gen'].env2_out_acc
                for tc in TC:
                        for subset in SUBSET:
                            # weak = weak_df[(weak_df.dataset==dataset)&(weak_df.subset==subset)&(weak_df.tc==int(tc))&(weak_df.model_selection=='last')].test_acc.item()
                            weak = weak_df[(weak_df.dataset==dataset)&(weak_df.subset==subset)&(weak_df.tc==tc)&(weak_df.model_selection=='last')].test_acc.item()
                            try:    
                                path = f'{dataset}/st{tail}/{seed}/{subset}_{tc}_{subset_st}_{st}'
                                #print(path)
                                # if st == 'b14' and dataset == 'Places365':
                                #     path = f'{dataset}/st/{seed}/0.5_{tc}_{subset_st}_{st}'
                                file = path + '/results.jsonl'
                                data = select_data(file)
                            except:
                                path = f'{dataset}/st{tail}/0/{subset}_{tc}_{subset_st}_{st}'
                                #print(path)
                                # if st == 'b14' and dataset == 'Places365':
                                #     path = f'{dataset}/st/0/0.5_{tc}_{subset_st}_{st}'
                                file = path + '/results.jsonl'
                                data = select_data(file)
                            for trait,values in data.items():
                                records.append(
                                    {
                                        'dataset':dataset,
                                        'tc':tc,
                                        'st':sts,
                                        'size':st,
                                        'model_selection':trait,
                                        'weak':weak,
                                        'subset':subset,
                                        'subset_st':subset_st,
                                        'ceiling':ceiling,
                                        'acc':values.env2_out_acc,
                                        'in_acc':values.env1_in_acc,
                                        'val_acc':values.env1_out_acc,
                                        'test_loss':values.env2_out_loss,
                                        'agreement':values.agreement,
                                        'agreement_neg':values.agreement_neg,
                                        'agreement_pos':values.agreement_pos,
                                        'step':int(values.step),
                                        'epoch':int(values.epoch),
                                        'pgr':(values.env2_out_acc-weak)/(ceiling-weak),
                                        'seed':seed
                                        #'cluster_err':values.cluster_err
                                    }
                                )
    return records 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--output_dir', type=str, default="all_acc")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ft',action='store_true')
    parser.add_argument('--dataset',type=str,default='Places365')
    parser.add_argument('--st',type=str,default='dinov2') #,choices=["dinov2","vit","mamba","50","dino","sbb","resnet-d","clip"]
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    #result = all_accs(ft=args.ft,twenty=args.twenty)
    records = all_accs_table(args.seed,args.ft,args.dataset,args.st)
    #file = output_dir+'/all_accs.json'
    # with open(file,'w') as f:
    #     json.dump(result,f)

    # Save the records as JSON
    file = output_dir+'/all_accs_table.json'
    with open(file, 'w') as json_file:
        json.dump(records, json_file, indent=4)
    print('done')
    #pprint.pprint(result)
    
