from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def _label_one_segment(client, system_prompt, model_name, segment_id, text_to_label):
    response = client.chat.completions.create(
        model=model_name,
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
        return segment_id, None
    return segment_id, clean_label(raw_label)

def _load_checkpoint(checkpoint_path):
    completed = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                completed[record["segment_id"]] = record["labels"]
        print(f"Resuming from checkpoint {checkpoint_path}: {len(completed)} segments already done.")
    return completed

def generate_labels_gpt(data_processed, system_prompt, config):
    api_key = os.getenv("OPENAI_API_KEY")
    client = config.get("_client") or OpenAI(
        api_key=api_key,
        base_url=config.get("base_url"),
        max_retries=config.get("max_retries", 5),
    )
    model_name = config["model_name"]
    max_concurrency = config.get("max_concurrency", 8)
    checkpoint_path = config.get("checkpoint_path")

    # segment_id -> labels (or None for irrelevant); pre-filled from checkpoint
    results = _load_checkpoint(checkpoint_path)
    pending = [k for k in data_processed.keys() if k not in results]

    checkpoint_file = None
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        checkpoint_file = open(checkpoint_path, "a", encoding="utf-8")

    failed = []
    try:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(_label_one_segment, client, system_prompt, model_name, k, data_processed[k]): k
                for k in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                k = futures[future]
                try:
                    segment_id, labels = future.result()
                except Exception as e:
                    failed.append(k)
                    print(f"Segment {k} failed: {e}")
                    continue
                results[segment_id] = labels
                if checkpoint_file:
                    checkpoint_file.write(json.dumps({"segment_id": segment_id, "labels": labels}) + "\n")
                    checkpoint_file.flush()
    finally:
        if checkpoint_file:
            checkpoint_file.close()

    if failed:
        print(f"{len(failed)} segments failed and were skipped: {failed}")

    # Assemble document-level output in segment_id order for determinism
    output = {}
    for k in sorted(results.keys()):
        labels = results[k]
        if labels is None or k not in data_processed:
            continue
        doc_dict = output.setdefault("_".join(k.split("_")[:-1]), {})
        annotation = {
            "sentence": data_processed[k],
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
