from openai import OpenAI
import uuid
import os
from tqdm import tqdm
import json

def clean_label(raw_label): # move to utils 
    labels = raw_label.split("\n")
    labels = [i.replace("LABEL: ", "").replace("[", "").replace("]", "").strip() for i in labels]
    return labels

def generate_labels(data_processed, system_prompt, config):
    if config is None:
        config = {}
    model_name = config["model_name"]
    if "gpt" in model_name.lower():
        return generate_labels_gpt(data_processed, system_prompt, config)
    # elif "llama" in model_name.lower():
    #     return generate_labels_hf(data_processed, system_prompt, config)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
def generate_labels_gpt(data_processed, system_prompt, config):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key = api_key)
    output = {}

    for k in tqdm(list(data_processed.keys())):
        text_to_label = data_processed[k]

        response = client.chat.completions.create(
            model=config["model_name"],
            messages=[
                {
                    "role": "developer", 
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": text_to_label
                }
            ],
            )
        raw_label = response.choices[0].message.content
        if "irrelevant" in raw_label.lower():
            continue
        else:
            labels = clean_label(raw_label)
            #Output to document-level
            doc_dict = output.setdefault("_".join(k.split("_")[:-1]),{})
            annotation = {
                "sentence": text_to_label,
                "label": labels,
            }
            doc_dict.setdefault("LLM_Annotation", []).append(annotation)
    return output

def save_generation_output(output, output_fpath_dir, output_id = None):
    if output_id is None:
        output_id = uuid.uuid1()
    if not os.path.exists(output_fpath_dir):
        os.makedirs(output_fpath_dir)
    output_fpath = os.path.join(output_fpath_dir, f"generation_{output_id}.json")
    print(f"Saving generation output to {output_fpath}")
    with open(output_fpath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    return output_id