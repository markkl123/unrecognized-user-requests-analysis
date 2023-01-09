import json
from sklearn.metrics import rand_score, adjusted_rand_score
from typing import Dict, List


def read_clusters(filename: str) -> (Dict[str, int], List[str]):
    """
    parsing clustering outcome and collecting cluster members
    :param filename: filename with the clustering outcome (json format)
    :return:
        item2cluster_id: dictionary of a request to its cluster id
        requests: a list of all clustered requests
    """

    item2cluster_id = dict()
    with open(filename, 'r', encoding='utf8') as json_file:
        solution = json.load(json_file)

    requests = list()
    for i, cluster in enumerate(solution['cluster_list']):
        for item in cluster['requests']:
            item2cluster_id[item] = i
        requests.extend(cluster['requests'])

    for item in solution['unclustered']:
        item2cluster_id[item] = -1
    requests.extend(solution['unclustered'])

    return item2cluster_id, requests


def evaluate_clustering(filename1: str, filename2: str):
    """
    parsing two clustering solutions and evaluation via rand and adjusted rand score
    :param filename1: clustering outcome #1
    :param filename2: clustering outcome #2
    """

    clusters1, requests1 = read_clusters(filename1)
    clusters2, requests2 = read_clusters(filename2)

    assert(len(requests1) == len(requests2))

    items = requests1  # set the items list (joint for both solutions)
    cluster_ids1 = [clusters1[i] for i in items]  # first  clusters assignments
    cluster_ids2 = [clusters2[i] for i in items]  # second clusters assignments

    print(f'rand score: {rand_score(cluster_ids1, cluster_ids2)}')
    print(f'adjusted rand score: {adjusted_rand_score(cluster_ids1, cluster_ids2)}')
    #print(len(requests1), len(requests2))
