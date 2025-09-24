import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer 
        
def get_sim_theme(gold_themes, pred_themes, embedding_model='all-MiniLM-L6-v2'):
    '''
    input: 
    gold_themes: data type is list, list of gold themes
    pred_themes: data type is list, list of predicted themes
    output:
    sim_dict: keys are all pred themes, values are pairs of (most similar gold theme for key, cos sim score between pred theme and most sim gold theme)

    for each pred_theme in pred_themes: find the most similar gold_theme and the cos sim score
    '''
    # Load embedding model
    embedding_model = SentenceTransformer(embedding_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Get embeddings for both lists
    pred_themes_embeddings = embedding_model.encode(pred_themes, 
                                                        convert_to_tensor=True, 
                                                        device=device)  # Shape: (len(pred_themes), embedding_dim)
    gold_themes_embeddings = embedding_model.encode(gold_themes, 
                                                         convert_to_tensor=True, 
                                                         device=device)  # Shape: (len(gold_themes), embedding_dim)

    # Normalize embeddings for cosine similarity
    pred_themes_embeddings = F.normalize(pred_themes_embeddings, p=2, dim=1)
    gold_themes_embeddings = F.normalize(gold_themes_embeddings, p=2, dim=1)


    # Compute cosine similarity between each gold_theme and all pred_themes
    cos_sim_matrix = torch.matmul(gold_themes_embeddings, pred_themes_embeddings.T)  # Shape: (len(gold_themes), len(pred_themes))

    return cos_sim_matrix
    # else:
    #     # Compute cosine similarity between each pred_theme and all gold_themes
    #     cos_sim_matrix = torch.matmul(pred_themes_embeddings, gold_themes_embeddings.T)  # Shape: (len(pred_themes), len(gold_themes))

    #     # Find the most similar gold_theme for each pred_theme
    #     sim_dict = {}
    #     for i, pred_theme in enumerate(pred_themes):
    #         best_match_idx = torch.argmax(cos_sim_matrix[i]).item()
    #         most_similar_gold_theme = gold_themes[best_match_idx]
    #         cos_sim_score = cos_sim_matrix[i, best_match_idx].item()
    #         sim_dict[pred_theme] = (most_similar_gold_theme, cos_sim_score)
    #     return sim_dict


def theme_precision(gold_themes, pred_themes, cos_sim_thresh=0.5):
    # calculate similarity
    sim_mat = get_sim_theme(gold_themes=gold_themes, pred_themes=pred_themes)
    sim_mat_thresh = torch.where(sim_mat >= cos_sim_thresh, sim_mat, float('nan'))
    
    true_positives_num = (~sim_mat_thresh.isnan().all(0)).sum().item()
    precision_score = true_positives_num / len(pred_themes)
    return precision_score

def theme_recall(gold_themes, pred_themes, cos_sim_thresh=0.5):
    # calculate similarity
    sim_mat = get_sim_theme(gold_themes=gold_themes, pred_themes=pred_themes)
    sim_mat_thresh = torch.where(sim_mat >= cos_sim_thresh, sim_mat, float('nan'))

    true_positives_num = (~sim_mat_thresh.isnan().all(1)).sum().item()
    recall_score = true_positives_num / len(gold_themes)
    return recall_score

def get_matched_pairs(gold_themes, pred_themes, cos_sim_thresh=0.5):
    sim_mat = get_sim_theme(gold_themes, pred_themes)
    above_thresh = torch.where(sim_mat >= cos_sim_thresh, sim_mat, float('nan'))
    # index of non-nan values
    above_thresh_indices = torch.nonzero(~torch.isnan(above_thresh))
    matched_pairs = []
    for idx in above_thresh_indices:
        matched = (gold_themes[idx[0]], pred_themes[idx[1]])
        matched_pairs.append(matched)
    return matched_pairs

def match_pred_to_gold(matched_pairs):
    pred_to_gold = {}
    for gold, pred in matched_pairs:
        pred_to_gold.setdefault(pred, []).append(gold)
    return pred_to_gold

def match_gold_to_pred(matched_pairs):
    gold_to_pred = {}
    for gold, pred in matched_pairs:
        gold_to_pred.setdefault(gold, []).append(pred)
    return gold_to_pred


def get_precision_by_theme(gold_themes, pred_themes, labeled_segments, similarity_threshold=0.5):
    """
    Params:
    gold_themes: list of gold themes
    pred_themes: list of predicted themes
    labeled_segments: dict, output of the label generation step with human annotations in the following format:
    {
        "doc1": {
                    "LLM_Annotation": [
                        {   "segment": "text of segment 1",
                            "theme": ["A", "B"], # final predicted themes after clustering
                            "gold_label": ["A"]
                        },
                        {   "segment": "text of segment 2",
                            "theme": ["C"],
                            "gold_label": ["B", "C"]
                        }
                    ]
                }
    }
    similarity_threshold: float, threshold for cosine similarity to consider a match, default is 0.5
    """
    matched_pairs = get_matched_pairs(gold_themes, pred_themes, similarity_threshold)
    pred_to_gold = match_pred_to_gold(matched_pairs)
    prec_by_theme = {}
    for doc in labeled_segments.keys():
        for segment in labeled_segments[doc]["LLM_Annotation"]:
            for pred_theme in segment['theme']:
                if pred_theme in list(pred_to_gold.keys()):
                    prec_by_theme.setdefault(pred_theme, {'tp':0, 'total':0})
                    prec_by_theme[pred_theme]['total'] += 1
                    # true positive if any of the gold labels match
                    for gold in segment['gold_label']:
                        if gold in pred_to_gold[pred_theme]:
                            prec_by_theme[pred_theme]['tp'] += 1
                            break
    return prec_by_theme

def segment_precision(prec_by_theme):
    # calculate total from prec_by_theme
    total_matched_seg = sum([v['total'] for v in prec_by_theme.values()])
    for v in prec_by_theme.values():
        v['precision'] = v['tp'] / v['total']
        v['weight'] = v['total'] / total_matched_seg
    # sum weighted precision
    seg_prec = sum([v['precision'] * v['weight'] for v in prec_by_theme.values()])
    return seg_prec
    
def get_recall_by_theme(gold_themes, pred_themes, labeled_segments, similarity_threshold=0.5):
    matched_pairs = get_matched_pairs(gold_themes, pred_themes, similarity_threshold)
    gold_to_pred = match_gold_to_pred(matched_pairs)
    recall_by_theme = {}
    for doc in labeled_segments.keys():
        for segment in labeled_segments[doc]["LLM_Annotation"]:
            for gold in segment['gold_label']:
                if gold in list(gold_to_pred.keys()): # gold label has matched predicted themes
                    
                    recall_by_theme.setdefault(gold, {'tp':0, 'total':0})
                    recall_by_theme[gold]['total'] += 1
                    # true positive if any of the pred themes match
                    for pred_theme in segment['theme']:
                        if pred_theme in gold_to_pred[gold]:
                            recall_by_theme[gold]['tp'] += 1
                            break
    return recall_by_theme

def segment_recall(recall_by_theme):
    # calculate total from recall_by_theme
    total_matched_seg = sum([v['total'] for v in recall_by_theme.values()])
    for v in recall_by_theme.values():
        v['recall'] = v['tp'] / v['total']
        v['weight'] = v['total'] / total_matched_seg
    # sum weighted recall
    seg_rec = sum([v['recall'] * v['weight'] for v in recall_by_theme.values()])
    return seg_rec
