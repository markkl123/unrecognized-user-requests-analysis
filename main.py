import json
from compare_clustering_solutions import evaluate_clustering
import numpy as np
from sklearn.cluster import KMeans


def x_means(data, k_min, k_max):
    best_k = k_min
    best_inertia = float('inf')

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)

        if kmeans.inertia_ < best_inertia:
            best_k = k
            best_inertia = kmeans.inertia_
            best_kmeans = kmeans

    return best_kmeans, best_k


# # Example usage
# X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# k_min = 2
# k_max = 5
# kmeans = x_means(X, k_min, k_max)
# print(kmeans.labels_)

def analyze_unrecognized_requests(data_file, output_file, num_rep, min_size):
    # todo: implement this function
    #  you are encouraged to break the functionality into multiple functions,
    #  but don't split your code into multiple *.py files
    #
    #  todo: the final outcome is the json file with clustering results saved as output_file
    print(data_file, output_file, num_rep, min_size)
    pass


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    analyze_unrecognized_requests(config['data_file'],
                                  config['output_file'],
                                  config['num_of_representatives'],
                                  config['min_cluster_size'])

    # todo: evaluate your clustering solution against the provided one
    evaluate_clustering(config['example_solution_file'], config['example_solution_file'])  # invocation example
    #evaluate_clustering(config['example_solution_file'], config['output_file'])
