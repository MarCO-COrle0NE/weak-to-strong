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
TC=["resnet18"]
SUBSET = [75,100,140,280,420,650]
#SUBSET_ST= [4550,5460,6370,6825,7280,8190,9100]
SUBSET_ST= [5460]
#ALGORITHM = ['Average']
#weak_performances = {'VLCS':[0.7381516588,0.75,0.8237559242],'PACS':[0.6452023416,0.7635530669,0.8796131331],'OfficeHome':[0.7075969704,0.786320863,0.8850126234]}
#weak_performances_avg = {'VLCS':[0.7390402844,0.7508886256,0.8279028436],'PACS':[0.6457113769,0.7632985492,0.8796131331],'OfficeHome':[0.7075969704,0.7867798944,0.8850126234]}
#cols_interest = [f'env{i}_in_acc' for i in range(4)]+[f'env{i}_out_acc' for i in range(4)]+['epoch','loss','step'] + ['env3_out_loss']#+['agreement','agreement_neg','agreement_pos']
#weak_df = pd.read_json('weak_subset/weak.json')

def select_data(file):
    df = pd.read_json(file,lines=True)
    # Find the index of the last occurrence of step=0
    #last_step_0_index = df[df['step'] == 0].index[-1]
    # Slice the DataFrame starting from the last step=0 row
    #df = df.iloc[last_step_0_index:]
    # Select models -- last
    model_last = df.iloc[-1]
    # Select models -- ground truth
    model_acc = df.loc[df['env1_out_acc'].idxmax()]
    # Select models -- source validation set
    model_loss = df.loc[df['env1_out_loss'].idxmin()]
    return {'last':model_last,'acc':model_acc,'loss':model_loss}
    
def all_accs_table(seed,ft=False,hundred=False):
    weak_df = pd.read_json(f'tc_ColoredMNISTID/{seed}/{TC[0]}/weak.json')
    # base_path = f'ColoredMNISTID/ceiling/1_resnet50'
    # base_file = base_path + '/results.jsonl'
    # ceiling_data = select_data(base_file)
    # ceiling = ceiling_data['acc'].env1_out_acc
    # ceiling_loss = ceiling_data['loss'].env1_out_loss
    records = []
    for dataset in DATASETS:
        for st in ST:
            for subset_st in SUBSET_ST:
                base_path = f'{dataset}/ceiling/{subset_st}_{st}'
                base_file = base_path + '/results.jsonl'
                ceiling_data = select_data(base_file)
                ceiling = ceiling_data['acc'].env1_out_acc
                ceiling_loss = ceiling_data['loss'].env1_out_loss
                for tc in TC:
                        for subset in SUBSET:
                            weak = weak_df[(weak_df.dataset==dataset)&(weak_df.subset==subset)&(weak_df.tc==tc)&(weak_df.model_selection=='last')].test_acc.item()
                            weak_loss = weak_df[(weak_df.dataset==dataset)&(weak_df.subset==subset)&(weak_df.tc==tc)&(weak_df.model_selection=='last')].test_loss.item()
                            if ft:
                                path = f'{dataset}/st_ft/{seed}/{subset}_{tc}_{subset_st}_{st}'
                            else:
                                path = f'{dataset}/st_resnet_hard/{seed}/{subset}_{tc}_{subset_st}_{st}'
                            #print(path)
                            file = path + '/results.jsonl'
                            data = select_data(file)
                            
                            for trait,values in data.items():
                                records.append(
                                    {
                                        'dataset':dataset,
                                        'tc':tc,
                                        'st':st,
                                        'model_selection':trait,
                                        'weak':weak,
                                        'weak_loss':weak_loss,
                                        'subset':subset,
                                        'subset_st':subset_st,
                                        'ceiling':ceiling,
                                        'ceiling_loss':ceiling_loss,
                                        'acc':values.env1_out_acc,
                                        'in_acc':values.env1_in_acc,
                                        'test_loss':values.env1_out_loss,
                                        'loss':values.loss,
                                        'agreement':values.agreement,
                                        'agreement_neg':values.agreement_neg,
                                        'agreement_pos':values.agreement_pos,
                                        'step':int(values.step),
                                        'epoch':int(values.epoch),
                                        'pgr_acc':(values.env1_out_acc-weak)/(ceiling-weak),
                                        'pgr_loss':(weak_loss-values.env1_out_loss)/(weak_loss-ceiling_loss),
                                        'cluster_err':values.cluster_err
                                    }
                                )
    return records 

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
    
