from tqdm import tqdm
import json
import argparse
import os


def is_truncated(axis_metadata):
    '''
    Rule check for truncated axis.
    '''
    if 'x' in axis_metadata['axis'] and 'y1' in axis_metadata['axis']:
        try:
            float_x_axis = [float(axis_metadata['label'][x]) for x in range(len(axis_metadata['label'])) if axis_metadata['axis'][x]=='x']
        except:
            float_x_axis = []
        try:
            float_y_axis = [float(axis_metadata['label'][y]) for y in range(len(axis_metadata['label'])) if axis_metadata['axis'][y]=='y1']
        except:
            float_y_axis = []    

        if len(float_x_axis)==0 or len([d for d in float_x_axis if (d>1800 and d < 2100)])== len(float_x_axis):
                #The x axis is categorical or contains only years range
                #We can check the y axis
                if len(float_y_axis) > 0:
                    y1_axis_first_tick = sorted(float_y_axis)[0]
                    if y1_axis_first_tick > 0: 
                        return True
        else:
            if len(float_y_axis)==0 or len([d for d in float_y_axis if (d>1800 and d < 2100)])== len(float_y_axis):
                #The y axis is categorical or contains only years range
                #We can check the x axis
                if len(float_x_axis) > 0:
                    x_axis_first_tick = sorted(float_x_axis)[0]
                    if x_axis_first_tick > 0: 
                        return True
    return False



def is_inverted(axis_metadata):
    '''
    Rule check for inverted axis.
    '''
    try:
        float_x_axis = [float(axis_metadata['label'][x]) for x in range(len(axis_metadata['label'])) if axis_metadata['axis'][x]=='x']
    except:
        float_x_axis = []
    try:
        float_y_axis = [float(axis_metadata['label'][y]) for y in range(len(axis_metadata['label'])) if axis_metadata['axis'][y]=='y1']
    except:
        float_y_axis = []   
    try:
        float_y2_axis = [float(axis_metadata['label'][y]) for y in range(len(axis_metadata['label'])) if axis_metadata['axis'][y]=='y2']
    except:
        float_y2_axis = []   


    #Y axis
    if len(float_y_axis) > 0 and not len([d for d in float_y_axis if (d>1800 and d < 2100)])== len(float_y_axis):
        if sorted(float_y_axis)==float_y_axis[::-1]:
            return True
    if len(float_y2_axis) > 0 and not len([d for d in float_y2_axis if (d>1800 and d < 2100)])== len(float_y2_axis):
        if sorted(float_y2_axis)==float_y2_axis[::-1]:
            return True
    #x axis
    if len(float_x_axis) > 0:
        if len([d for d in float_x_axis if (d>1800 and d < 2100)])== len(float_x_axis):
            #Ordinal dates on the x axis
            if sorted(float_x_axis)==float_x_axis[::-1]:
                return True
        #Numerical values on the x axis
        if sorted(float_x_axis)==float_x_axis[::-1]:
            return True
    return False


def is_inconsistent_tick(axis_metadata):
    '''
    Rule check for inconsitent tick intervals.
    '''
    for axis in ['x', 'y1', 'y2']:
        try:
            label = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]==axis]
            position = [float(axis_metadata['relative_position'][d]) for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]==axis]
            label = [int(label[i]) if type(label[i])==str else label[i] for i in range(len(label))]
            if sorted(label)==label or sorted(label)==label[::-1]:
                #the inconsistent ticks are not due to random shuffling of the values
                intervals = [round(label[i+1] - label[i],3) for i in range(len(label)-1)]
                if len(list(set(intervals))) > 1:
                    return True
            #Check for relative positions
            position_lengths = [len(str(position[i]).split('.')[1]) for i in range(len(position))]
            if len([p for p in position_lengths if p <= 2 ]) ==len(position):
                intervals = [round(position[i+1] - position[i],3) for i in range(len(position)-1)]
                if len(list(set(intervals))) > 1:
                    return True
        except:
            pass

    return False


def is_inconsistent_binning(axis_metadata):
    '''
    Rule check for inconsistent binning size.
    '''
    if 'x' in axis_metadata['axis']:
        try:
            x_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='x']
            if '-' in x_axis[0]:
                x_axis = [x.split('-') for x in x_axis] 
                intervals = [int(x[1])-int(x[0]) for x in x_axis]
            elif 'to' in x_axis[0].lower():
                x_axis = [x.lower().split('to') for x in x_axis] 
                intervals = [int(x[1])-int(x[0]) for x in x_axis]
            else:
                return False
            
            if len(list(set(intervals)))!=1:
                return True
        except:
            pass
    if 'y1' in axis_metadata['axis']:
        try:
            y_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y1']
            if '-' in y_axis[0]:
                y_axis = [y.split('-') for y in y_axis] 
                intervals = [int(y[1])-int(y[0]) for y in y_axis]
            elif 'to' in y_axis[0].lower():
                y_axis = [y.lower().split('to') for y in y_axis] 
                intervals = [int(y[1])-int(y[0]) for y in y_axis]
            else:
                return False
            
            if len(list(set(intervals)))!=1:
                return True
        except:
            pass
    return False


def is_inappropriate_order(axis_metadata):
    '''
    Rule check for inappropriate item order (one specific case: dates are not chronological on the axis).
    '''
    try:
        float_x_axis = [float(axis_metadata['label'][x]) for x in range(len(axis_metadata['label'])) if axis_metadata['axis'][x]=='x']
    except:
        float_x_axis = []
    try:
        float_y_axis = [float(axis_metadata['label'][y]) for y in range(len(axis_metadata['label'])) if axis_metadata['axis'][y]=='y1']
    except:
        float_y_axis = []  
    if len([d for d in float_x_axis if (d>1800 and d < 2100)])== len(float_x_axis) and len(float_x_axis) > 0:
        #The X axis is made of dates
        #They need to be in increasing or decreasing order but not random (otherwise it is inappropriate item order)
        if not sorted(float_x_axis) == float_x_axis and not sorted(float_x_axis)==float_x_axis[::-1]:
            return True
    if len([d for d in float_y_axis if (d>1800 and d < 2100)])== len(float_y_axis) and len(float_y_axis) > 0:
        #The Y axis is made of dates
        #They need to be in increasing or decreasing order but not random (otherwise it is inappropriate item order)
        if not sorted(float_y_axis) == float_y_axis and not sorted(float_y_axis)==float_y_axis[::-1]:
            return True

    return False
    

def is_dual(axis_metadata):
    '''
    Rule check for dual axis
    '''
    if 'y1' in axis_metadata['axis']:
        y1_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y1']
    else:
        return False
    if 'y2' in axis_metadata['axis']:
        y2_axis = [axis_metadata['label'][d] for  d in range(len(axis_metadata['label'])) if axis_metadata['axis'][d]=='y2']
    else:
        return False
    if y1_axis != y2_axis:
        #the two axis ticks differ
        return True
    else:
        return False
    

def coerce_number(value):
    try:
        return int(value)        
    except ValueError:
        try:
            return float(value)    
        except ValueError:
            return value          


def process_axis_data(axis_str):
    #Convert the axis extracted by the fine-tuned DePlot model to a dictionary format that can be used by the linter    
    if '\n' in axis_str:
        axis = axis_str.split('\n')[1:]
    else:
        axis = axis_str.split('<0x0A>')[1:]
    axis_dict = {'axis': [], 'label': [], 'Relative position': []}
    for item in axis:
        try:
            _, ax, la, rel = item.split(' | ')
            axis_dict['axis'].append(ax.strip())
            axis_dict['label'].append(coerce_number(la.strip().replace('%','')))
            axis_dict['Relative position'].append(rel.strip())
        except:
            pass

    return axis_dict


def get_linter_predictions(dataset='misviz', split = 'test', predicted=False):
    
    axis_path = f"src/output/predicted_axis_misviz/merged_axis_data_{dataset}.json"
    metadata = [m for m in json.load(open(f"data/{args.dataset}/{args.dataset}.json", encoding="utf-8")) if m['split']==split]
    axis_data = json.load(open(axis_path, encoding="utf-8"))[split]
   
    if dataset=='misviz_synth' and not predicted:
        #use the ground truth axis
        axis_data_path = [metadata[m]['axis_data_path'] for m in range(len(metadata))]
        axis_metadata = []
        for a in tqdm(range(len(axis_data_path))):
            axis_metadata.append(open(json.load(f"data/misviz_synth_V2/vis_output/{axis_data_path[a]}"),'r'))                                                  
    else:
        axis_metadata = [process_axis_data(a) for a in axis_data]
        
    predictions = []

    for m in tqdm(range(len(axis_metadata))):
        pred = []
        if is_inverted(axis_metadata[m]):
            pred.append('inverted axis')

        if is_truncated(axis_metadata[m]):
            pred.append('truncated axis')

        if is_inconsistent_tick(axis_metadata[m]):
            pred.append('inconsistent tick intervals')

        if is_inconsistent_binning(axis_metadata[m]):  
            pred.append('inconsistent binning size')

        if is_dual(axis_metadata[m]):
            pred.append('dual axis')

        if is_inappropriate_order(axis_metadata[m]):
            pred.append('inappropriate item order')

        if len(pred)==0:
            #No misleader check was passed
            pred = ['no misleader']
        predictions.append(pred)

    results = []
    for p in range(len(predictions)):
        results.append(
            {
            'image_path': metadata[p]['image_path'],
            'predicted_misleader': ','.join(predictions[p]),
            'true_misleader': metadata[p]['misleader']
            }
        )
    return results

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='misviz',  help="Dataset to use")
    parser.add_argument('--split', type=str, default='test', help='the dataset split to use')
    parser.add_argument('--use_predicted_axis',type=int, default=0, help='If 1, use the predicted axis. Otherwise, use the ground truth one.')
    args = parser.parse_args()


    results = get_linter_predictions(args.dataset, split=args.split, predicted=args.use_predicted_axis)
    os.makedirs('results', exist_ok=True)
    if args.use_predicted_axis:
        os.makedirs('results/linter', exist_ok=True)
        output_file = f'results/linter/{args.dataset}_{args.split}.json'
    else:
        os.makedirs('results/linter_gt', exist_ok=True)
        output_file = f'results/linter_gt/{args.dataset}_{args.split}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)