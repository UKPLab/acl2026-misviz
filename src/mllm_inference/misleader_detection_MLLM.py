from tqdm import tqdm
from utils import *
from llm_inference import *
from loaders import *
import transformers
import argparse
import os
import json


transformers.set_seed(42)



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='misviz',  help="Dataset to use")
    parser.add_argument('--split', type=str, default='test', help='the dataset split to use')
    parser.add_argument('--model', type=str, required=True,  help="Name of the model to run")
    parser.add_argument('--max_tokens', type=int, default=200,  help="Max number of generated tokens")
    parser.add_argument('--predict_bbox', type=int, default=0, help='set to 1 to predict bbox coordinates in addition to the misleaders')



    args = parser.parse_args()
    m = args.model
    max_tokens = args.max_tokens
    print(f'Generating detection answers for model {m}')
    #Load model
    if m in ['GPT41', 'o3', 'gemini-2.5-flash-lite']:
        model, tokenizer = m, ''
    else:
        # pass
        model, tokenizer = load_model(m)
    print('Model loaded')



    os.makedirs('results',exist_ok=True)
    root = f'results/{"-".join(args.model.split("/"))[:-1]}/'
    os.makedirs(root,exist_ok=True)

    #Misleaders set
    model_answers = []  
    dataset = [d for d in json.load(open(f"data/{args.dataset}/{args.dataset}.json", encoding="utf-8")) if d['split'] == args.split]
        
    for d in tqdm(range(len(dataset))):


        image_path = dataset[d]['image_path']
        if args.dataset=='misviz':
            image_path = f"data/misviz/{image_path}"
        else:
            image_path = f"data/misviz_synth/{image_path}"


        if not args.predict_bbox:
            prompt = """
            You are an expert in data visualization analysis. Your task is to identify misleaders present in the given visualization.

            Please carefully examine the visualization and detect its misleaders. Provide all relevant misleaders, up to three, as a comma separated list.
            In most cases only one misleader is relevant.
            If you detect none of the above types of misleaders in the visualization, respond with "no misleader".

            The available misleaders to select are, by alphabetical order:
            - discretized continuous variable: a map displays a continuous variable is transformed into a categorical variable by cutting it into discrete categories, thus exaggerating the difference between boundary cases.
            - dual axis: there are two independent y-axis, one on the left and one on the right, with different scales.
            - inappropriate axis range: the axis range is too broad or too narrow.
            - inappropriate item order: instances of a variable along an axis are in an unconventional, non-linear or non-chronological order.
            - inappropriate use of line chart: a line chart is used in inappropriate or unconventional ways, e.g., using a line chart with categorical variables, or encoding the time dimension on the y-axis.   
            - inappropriate use of pie chart: a pie chart does not display data in a part-to-whole relationship, e.g., its shares do not sum to 100%.
            - inconsistent binning size: a variable, such as years or ages, is grouped in unevenly sized bins.
            - inconsistent tick intervals: the tick values in one axis are not evenly spaced, e.g., the tick value sequence is 10, 20, 40, 45.
            - inverted axis: an axis is displayed in a direction opposite to conventions, e.g., the y-axis displays values increasing from top to bottom or the x-axis displays values increasing from right to left.
            - misrepresentation:  the value labels displayed do not match the size of their visual encodings, e.g., bars may be drawn disproportionate to the corresponding numerical value.
            - truncated axis: an axis does not start from zero, resulting in a visual exaggeration of changes in the dependent variable with respect to the independent variable. 
            - 3d: the visualization includes three-dimensional effects.

            Provide only the final answer, without additional explanation.
            """
        else:
            img_dim = Image.open(image_path).size
            prompt =  f"""You are given a chart (dimensions: {img_dim[0]} x {img_dim[1]}) with potential misleading regions:
                Please analyze the image to detect misleaders and define bounding box coordinates for any misleading regions.
                ** Let’s think it step by step! ** 
                Here is the list of potential misleaders and their corresponding misleading regiions:
                - discretized continuous variable: a map displays a continuous variable is transformed into a categorical variable by cutting it into discrete categories, thus exaggerating the difference between boundary cases. The misleading region is the legend of the map.
                - dual axis: there are two independent y-axis, one on the left and one on the right, with different scales. The misleading regions are the two vertical axes.
                - inappropriate axis range: the axis range is too broad or too narrow. The misleading region is the vertical axis.
                - inappropriate item order: instances of a variable along an axis are in an unconventional, non-linear or non-chronological order. The misleading region is (parts of) an axis.
                - inappropriate use of line chart: a line chart is used in inappropriate or unconventional ways, e.g., using a line chart with categorical variables, or encoding the time dimension on the y-axis.  The misleading region is one of the axis. 
                - inappropriate use of pie chart: a pie chart does not display data in a part-to-whole relationship, e.g., its shares do not sum to 100%. The misleading region is the labels on the pie slices.
                - inconsistent binning size: a variable, such as years or ages, is grouped in unevenly sized bins. The misleading region is one of the axis or the legend for a map.
                - inconsistent tick intervals: the tick values in one axis are not evenly spaced, e.g., the tick value sequence is 10, 20, 40, 45. The misleading region is (parts of) one of the axis.
                - inverted axis: an axis is displayed in a direction opposite to conventions, e.g., the y-axis displays values increasing from top to bottom or the x-axis displays values increasing from right to left. The misleading region is one of the axis.
                - misrepresentation:  the value labels displayed do not match the size of their visual encodings, e.g., bars may be drawn disproportionate to the corresponding numerical value. The misleading region involves at least two objects (bar, pie slice) for which the size difference is not proportional to the value label difference.
                - truncated axis: an axis does not start from zero, resulting in a visual exaggeration of changes in the dependent variable with respect to the independent variable. The misleading region is the starting tick of the axis.
                - 3d: the visualization includes three-dimensional effects. The misleading region is the 3D area of the chart.
                Then output a JSON file containing coordinates for the potential misleaders and explanations.
                *** Instructions:
                - ** Please analyze the image (dimensions: {img_dim[0]} x {img_dim[1]}) to detect any misleading regions.**
            
                - **Provide the misleading region coordinates with the name of the corresponding misleader**
                    - Your response format must strictly follow
                    the example JSON format:
                    ```
                    [
                    {{"coordinates": [[100, 200], [150, 200],[100, 300], [150, 300]],"misleader": "Truncated axis"}},
                    {{"coordinates": [[250, 300], [300, 300],[250, 350], [300, 350]], "misleader": "Misrepresentation"}}]
                    ```  
                """

        predicted_answer = generate_answer(image_path, 
                                        prompt, 
                                        tokenizer, 
                                        model, 
                                        m, 
                                        max_tokens)

        if m in ['GPT41', 'o3', 'gemini-2.5-flash-lite']:
            usage = predicted_answer[1] 
            predicted_answer = predicted_answer[0]
        else:
            usage=''
        model_answers.append({'image_path':dataset[d]['image_path'], 
                            'predicted_misleader':predicted_answer,
                            'true_misleader': dataset[d]['misleader'],
                            'chart_type': dataset[d]['chart_type'],
                            'usage': usage})

    if not args.predict_bbox:
        output_path = f'{args.dataset}_{args.split}.json'
    else:
        output_path = f'{args.dataset}_{args.split}_bbox.json'
    with open(os.path.join(root, output_path), 'w') as json_file:
                json.dump(model_answers, json_file, indent=4)