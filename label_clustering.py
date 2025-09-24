from openai import OpenAI
from tqdm import tqdm
import json
import os
import uuid

def process_labels(data):
    labels = []
    for k in data.keys():
        for seg in data[k]['LLM_Annotation']:
            labels += seg['label']
    labels = list(set(labels))
    return labels


def make_clustering_prompt(goal=None, dataset=None):
    if dataset is None and goal is not None:
        raise ValueError("If dataset is None, the description of the goal of inductive coding must be provided.")
    if dataset is not None:
        if dataset == "mediacorpus":
            goal = "understanding if there exists a generalizable framing dimension to influence us with decisions on policy issues."
        elif dataset == "astrobot":
            goal = "understanding the query type to the literature search bot from the astronomy scientists."
        elif dataset == "emotions":
            goal = "understanding the specific research motivations of the given paper about emotion recognition."
        elif dataset == "values":
            goal = "identifying the values of this machine learning research."
        elif dataset == "salescontest":
            goal = "identify what specific types of sales strategies or techniques used to drive Opioid sales."
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    system_prompt = f"""
    Synthesize the entire list of labels by clustering similar labels that are inductively labeled. 
    The clustering is to finalize MEANINGFUL and INSIGHTFUL THEMES for {goal}
    Output in json format where the key is the cluster, and the value is the list of input labels in that cluster. 
    For each cluster, the value should only take labels from the user input.
    ONLY output the JSON object, and do not add any other text.
    """ 

    return system_prompt

def save_iteration(iteration_result, n_iter, dataset, cluster_model_name, generation_model_name, output_dir):
    result_dir = os.path.join(output_dir, cluster_model_name, f"base-model_{generation_model_name}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, f"cluster_iter_{n_iter}.json"), "w") as f:
        json.dump(iteration_result, f, indent=4)

def _run_batch(client, system_prompt, cluster_model_name, user_input):
    response = client.chat.completions.create(
        model = cluster_model_name,
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": system_prompt
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": user_input
                }
            ]
            }
        ],
        response_format={
            "type": "json_object"
        },
        temperature=0,
        max_completion_tokens=8192,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        
    try:
        model_output = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        model_output = {}
    return model_output
            
def cluster_labels_gpt(generation_result, system_prompt, config, save_intermediate=True, gen_result_id=None, max_n_iter=3):
    if type(generation_result) is str:
        with open(generation_result, "r") as f:
            generation_result = json.load(f)

    labels_to_cluster = process_labels(generation_result)
    # Param
    cluster_model_name = config["cluster_model_name"]
    max_n_iter = config["max_n_iter"] if "max_n_iter" in config else max_n_iter

    print(f"========Run {cluster_model_name} Clustering=========")
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    if gen_result_id is None:
        print("The output id of label generation is not provided. The clustering output will use a new uuid.")
        output_id = uuid.uuid1()
    else:
        output_id = gen_result_id
    cluster_list = []
    for i in range(max_n_iter):
        batch_size = 100
        n_batch = len(labels_to_cluster) // batch_size if len(labels_to_cluster) % batch_size == 0 else len(labels_to_cluster) // batch_size + 1
        print(f"Number of batches: {n_batch}")
        # run clustering by batch
        cluster = {}
        for b in tqdm(range(n_batch)):
            user_input = str(labels_to_cluster[b*batch_size:(b+1)*batch_size])
            model_output = _run_batch(client, system_prompt, cluster_model_name, user_input)
            if model_output != {}:
                for k in model_output.keys():
                    cluster.setdefault(k, []).extend(model_output[k])
        labels_to_cluster = list(cluster.keys())
        cluster_list.append(cluster)
        # save intermediate results
        if save_intermediate:
            save_iteration(cluster_result = cluster, n_iter = i, 
                           config = config, output_id = output_id)
            print(f"Saved iteration {i} clustering result to {os.path.join(config['cluster_output_dir'], f'clustering_{output_id}', f'cluster_iter_{i}.json')}")
        if n_batch <= 1:
            break
    return cluster_list

def save_iteration(cluster_result, n_iter, config, output_id):
    result_dir = os.path.join(config["cluster_output_dir"], f"clustering_{output_id}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, f"cluster_iter_{n_iter}.json"), "w") as f:
        json.dump(cluster_result, f, indent=4)