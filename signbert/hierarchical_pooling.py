import torch
import torch.nn.functional as F


def hierarchical_pooling(node_features, subsets):
    pooled_features = []
    for subset in subsets:
        # Извлекаем узлы, относящиеся к текущему подмножеству
        subset_features = node_features[subset]
        # Max poooling
        pooled_feature = torch.max(subset_features, dim=0)[0]
        pooled_features.append(pooled_feature)

    final_feature = torch.max(pooled_features, dim=0)[0]
    return final_feature
