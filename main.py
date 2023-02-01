import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from compare_clustering_solutions import evaluate_clustering
import hdbscan



def cluster_requests(requests, min_size):
    # Convert requests to embeddings
    embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(requests)

    # Perform clustering
    clusters = np.array(hdbscan.HDBSCAN(min_cluster_size=8, metric='euclidean').fit(embeddings).labels_)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)

    # Remove clusters that are too small
    for cluster_num in range(n_clusters):
        cluster_mask = clusters == cluster_num

        if cluster_mask.sum() < min_size:
            clusters[cluster_mask] = -1

    return embeddings, clusters


def extract_cluster_representatives(requests, embeddings, clusters, num_rep):
    # todo: implement representatives extraction
    return {cluster_num: [f'rep{i}' for i in range(num_rep)] for cluster_num in list(set(clusters[clusters > -1]))}


def construct_cluster_names(requests, embeddings, clusters, rep_dict):
    # todo: implement names construction
    return {cluster_num: f'name{cluster_num}' for cluster_num in list(set(clusters[clusters > -1]))}


def save_cluster_results(output_file, requests, clusters, rep_dict, names_dict):
    results = {
        "cluster_list": [
            {
                "cluster_name": name,
                "representative_sentences": reps,
                "requests": requests[clusters == cluster_num].tolist()
            }
            for (cluster_num, reps), name in zip(rep_dict.items(), names_dict.values())
        ],
        "unclustered": requests[clusters == -1].tolist()
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):

    print(f'start analyzing unrecognized requests from {data_file}')
    requests = pd.read_csv(data_file)['request'].to_numpy()

    print(f'cluster {len(requests)} requests, with minimum of {min_size} requests per cluster')
    embeddings, clusters = cluster_requests(requests, min_size)

    print(f'choose {num_rep} representatives for each cluster')
    rep_dict = extract_cluster_representatives(requests, embeddings, clusters, num_rep)

    print(f'construct a meaningful name for each cluster')
    names_dict = construct_cluster_names(requests, embeddings, clusters, rep_dict)

    print(f'save results in {output_file}')
    save_cluster_results(output_file, requests, clusters, rep_dict, names_dict)


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # evaluate clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['output_file'])
