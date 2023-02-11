import json
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from transformers import pipeline, set_seed
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import community_detection

from compare_clustering_solutions import evaluate_clustering


def cluster_requests(all_requests, min_size):
    print(f'cluster {len(all_requests)} requests, with minimum of {min_size} requests per cluster')

    # Convert requests to embeddings
    embeddings = SentenceTransformer('all-MiniLM-L6-v2').encode(all_requests, convert_to_tensor=True)

    # Perform clustering
    grouped_clusters = community_detection(embeddings, min_community_size=min_size, threshold=0.631)

    # Construct cluster id's list
    clusters = np.full(len(embeddings), -1)
    for cluster_id, embedding_ids in enumerate(grouped_clusters, start=1):
        clusters[embedding_ids] = cluster_id

    return embeddings, clusters


def extract_representatives(requests, embeddings, num_rep):
    """
        TODO: options to consider
        1. transform embeddings
        2. select according to dimensions
    """
    components = PCA(n_components=num_rep).fit(embeddings).components_

    return [
        requests[np.argmin(np.linalg.norm(embeddings - component, axis=1))]
        for component in components
    ]


def extract_cluster_representatives(all_requests, all_embeddings, all_clusters, num_rep):
    print(f'choose {num_rep} representatives for each cluster')

    return {
        cluster_num: extract_representatives(all_requests[all_clusters == cluster_num],
                                             all_embeddings[all_clusters == cluster_num],
                                             num_rep)
        for cluster_num in list(set(all_clusters[all_clusters > -1]))
    }


# Load the summarization model from the pipeline
summarization_model = pipeline("summarization", model='facebook/bart-large-cnn')


def construct_name(requests: list[str]):
    article = ''

    def preprocess(text):
        result = []
        for token in simple_preprocess(text):
            if token not in STOPWORDS and len(token) > 2:
                result.append(token)
        return ' '.join(result)

    for s in requests:
        article += (preprocess(s) + '. ')

    # Set the seed for reproducibility
    set_seed(42)

    # Generate a summary for the article text
    title = summarization_model(article[:1024], max_length=11, min_length=4, early_stopping=True)[0].get("summary_text")
    title = title.split('?')[0]
    title = title.split('.')[0]
    print(title)
 
    return title


def construct_cluster_names(all_requests, all_clusters):
    print(f'construct a meaningful name for each cluster')

    return {
        cluster_num: construct_name(all_requests[all_clusters == cluster_num])
        for cluster_num in list(set(all_clusters[all_clusters > -1]))
    }


def save_cluster_results(output_file, requests, clusters, rep_dict, names_dict):
    print(f'save results in {output_file}')

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

    # Read all requests
    requests = pd.read_csv(data_file)['request'].to_numpy()

    # First, try cluster the requests
    embeddings, clusters = cluster_requests(requests, min_size)

    # Then, for each cluster extract the most representative instances
    rep_dict = extract_cluster_representatives(requests, embeddings, clusters, num_rep)

    # Finally, give each cluster a meaning full name
    names_dict = construct_cluster_names(requests, clusters)

    # Store the results
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
