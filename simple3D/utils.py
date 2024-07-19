import numpy as np

def generate_neighbors_and_distances(num_labels_per_group=100, probability_creating_distance=0.15,
                                     seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Generate labels for two groups
    group1_labels = np.arange(num_labels_per_group)
    group2_labels = np.arange(num_labels_per_group, 2*num_labels_per_group)
    
    # Generate neighbors list within each group
    neighbors = []
    for i in range(num_labels_per_group - 1):
        neighbors.append((group1_labels[i], group1_labels[i+1]))  # Group 1 neighbors
    for i in range(num_labels_per_group - 1):
        neighbors.append((group2_labels[i], group2_labels[i+1]))  # Group 2 neighbors
    
    # Generate distances
    desired_distances = []
    for i in range(num_labels_per_group):
        if np.random.rand() > probability_creating_distance: 
            continue
        for j in range(i + 1, num_labels_per_group):
            if np.random.rand() > probability_creating_distance: 
                continue
            desired_distances.append((i, j, np.random.uniform(1.0, min(abs(i-j), 10.0))))

    for i in range(num_labels_per_group, num_labels_per_group * 2):
        if np.random.rand() > probability_creating_distance: 
            continue
        for j in range(i + 1, num_labels_per_group * 2):
            if np.random.rand() > probability_creating_distance: 
                continue
            desired_distances.append((i, j, np.random.uniform(1.0, min(abs(i-j), 10.0))))

    probability_creating_distance = probability_creating_distance * 0.6
    for i in range(num_labels_per_group):
        if np.random.rand() > probability_creating_distance: 
            continue
        for j in range(num_labels_per_group, num_labels_per_group * 2):
            if np.random.rand() > probability_creating_distance: 
                continue
            desired_distances.append((i, j, np.random.uniform(5.0, 20.0)))
    
    return neighbors, desired_distances

def main():

    # Example usage:
    neighbors, desired_distances = generate_neighbors_and_distances(num_labels_per_group=80, probability_creating_distance=0.1)

    print(f"Neighbors:\n{neighbors[:10]}")  # Print first 10 neighbors for brevity
    print(f"Desired distances (triangular inequalities):\n{desired_distances[:10]}")  # Print first 10 desired distances

if __name__ == "__main__":
    main()