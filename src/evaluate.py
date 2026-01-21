import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import json 


def post_process_pred(pred):
    #Convert the predictions to a list
    if 'no misleader' in pred.lower():
        pred = []
    elif ',' in pred:
        pred = pred.lower().replace('\n','').split(',')
    else:
        pred = [pred.lower().replace('\n','')]
    return pred

def compute_metrics(data):
    #binary metrics
    binary_predictions = ['misleading' if len(post_process_pred(d['predicted_misleader']))>=1 else 'non-misleading' for d in data]
    binary_ground_truths = ['misleading' if len(d['true_misleader'])>=1 else 'non-misleading' for d in data]
    acc = round(100*accuracy_score(binary_ground_truths, binary_predictions),1)
    f1 = round(100*f1_score(binary_ground_truths, binary_predictions, average='macro'),1)
    precision = round(100*precision_score(binary_ground_truths, binary_predictions,  pos_label='misleading'),1)
    recall = round(100*recall_score(binary_ground_truths, binary_predictions, pos_label='misleading'),1)
    #multiclass metrics
    PM = []
    EM = []
    for d in data:
        pred = set(post_process_pred(d['predicted_misleader']))
        gt = set(d['true_misleader'])
        #Only compute this for the instances with misleaders
        if len(gt) >=1:
            if len(pred)==0:
                EM.append(0)
                PM.append(0)
            elif pred==gt:
                EM.append(1)
                PM.append(1)
            elif pred.issubset(gt): 
                EM.append(0)
                PM.append(1)
            else:
                EM.append(0)
                PM.append(0)


    EM = round(100*sum(EM)/len(EM), 2)
    PM = round(100*sum(PM)/len(PM), 2)

    
    print(f"Accuracy: {acc} | Precision: {precision} | Recall: {recall} | F1: {f1} | Exact match misleader {EM} | Partial match misleader {PM}")
    return acc, precision, recall, f1, EM, PM



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='misviz',  help="Dataset to use")
    parser.add_argument('--split', type=str, default='test', help='the dataset split to use')
    parser.add_argument('--model', type=str, required=True,  help="Name of the model to run")
    args = parser.parse_args()

    print('-----------------------------------------')
    print(f"{args.model} - {args.dataset}")
    results = json.load(open(f"results/{args.model}/{args.dataset}_{args.split}.json", encoding="utf-8"))
    compute_metrics(results)
    print('-----------------------------------------')