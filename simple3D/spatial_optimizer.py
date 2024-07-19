import sys
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp

class SpatialOptimizer:
    def __init__(self, N, neighbor_pairs, desired_distances, 
                 upper_threshold=5.0, cube_size=10):
        self.N = N
        self.neighbor_pairs = neighbor_pairs
        self.desired_distances = desired_distances
        self.upper_threshold = upper_threshold
        self.strong_restraint_weight = 10.0
        self.weak_restraint_weight   = 0.05
        self.mid_restraint_weight    = 5.00
        self.cube_size = cube_size
        self.models = None
        self.completed_processes = 0

    def strong_restraint(self, coords):
        """
        To enforce that neighbouring particles are togther in 3D.
        """
        coords = coords.reshape(self.N, 3)
        cost = 0
        for i, j in self.neighbor_pairs:
            cost += np.sum((coords[i] - coords[j])**2)
        return cost

    def weak_restraint(self, coords):
        """
        To spread out the 3D model and prevent collapsing.
        """
        coords = coords.reshape(self.N, 3)
        cost = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < self.upper_threshold:
                    if dist < 1e-3:  # Small threshold to avoid division by zero
                        dist = 1e-3
                    cost += 1 / dist
        return cost

    def mid_restraint(self, coords):
        """
        Attractive force between particles with observed contacts.
        """
        coords = coords.reshape(self.N, 3)
        cost = 0
        for i, j, d in self.desired_distances:
            cost += (np.linalg.norm(coords[i] - coords[j]) - d)**2
        return cost

    def objective_function(self, coords):
        return (
            self.strong_restraint_weight * self.strong_restraint(coords) +
            self.weak_restraint_weight   * self.weak_restraint(coords) +
            self.mid_restraint_weight    * self.mid_restraint(coords)
        )

    def objective_gradient(self, coords):
        coords = coords.reshape(self.N, 3)
        grad = np.zeros_like(coords)
        strong_weight = self.strong_restraint_weight * 2
        weak_weight   = self.weak_restraint_weight   * 2
        mid_weight    = self.mid_restraint_weight    * 2

        # Compute gradient for strong restraint
        for i, j in self.neighbor_pairs:
            diff = coords[i] - coords[j]
            tmp = strong_weight * diff
            grad[i] += tmp
            grad[j] -= tmp

        # Compute gradient for weak restraint
        for i in range(self.N):
            for j in range(i + 1, self.N):
                diff = coords[i] - coords[j]
                dist = np.linalg.norm(diff)
                if dist < self.upper_threshold:
                    if dist < 1e-3:  # Small threshold to avoid division by zero
                        dist = 1e-3
                    tmp = weak_weight * diff / (dist**3)
                    grad[i] -= tmp
                    grad[j] += tmp

        # Compute gradient for mid restraint
        for i, j, d in self.desired_distances:
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist > 0:  # To avoid division by zero
                tmp = mid_weight * (dist - d) * diff / dist
                grad[i] += tmp
                grad[j] -= tmp

        return grad.flatten()

    def generate_initial_guess(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return (np.random.rand(self.N, 3).flatten()- 0.5) * self.cube_size / 4

    def run_single_optimization(self, method='L-BFGS-B', seed=None):
        bounds = [(-self.cube_size / 2, self.cube_size / 2) for _ in range(self.N * 3)]
        initial_guess = self.generate_initial_guess(seed)
        
        if method == 'L-BFGS-B':
            options = {
               'maxls': 50  # Increase the number of line searches
               }
        else:
            options={}
        
        result = minimize(self.objective_function, initial_guess,
                          jac=self.objective_gradient,
                          bounds=bounds, method=method, options=options)
        result.x = result.x.reshape((self.N, 3))
        return result


    # Callback function to update the status
    def update_status(self, _):
        self.completed_processes += 1
        sys.stdout.write(f"\rCompleted processes: {self.completed_processes:>8,}")
        sys.stdout.flush()

    # Function to run multiple optimizations in parallel with status updates
    def run_optimizations_parallel(self, num_iterations, method='L-BFGS-B',
                                   n_cpus=None):
        self.completed_processes = 0
        if n_cpus is None:
            n_cpus = mp.cpu_count()
        pool = mp.Pool(n_cpus)
        results = [pool.apply_async(self.run_single_optimization, args=(method, seed),
                                    callback=self.update_status) for seed in range(num_iterations)]
        pool.close()
        pool.join()
        self.models = [r.get() for r in results]
        # self.models = [r for r in self.models if r.success]

    def view(self, model_number, show_restraints=True, show_labels=True, ax=None):
        
        final_coordinates = self.models[model_number].x
        
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot points
        ax.scatter(final_coordinates[:, 0],
                final_coordinates[:, 1],
                final_coordinates[:, 2], c='grey', marker='o', s=2, alpha=0.5)

        # Annotate points
        if show_labels:
            for i in range(self.N):
                ax.text(final_coordinates[i, 0], final_coordinates[i, 1], 
                        final_coordinates[i, 2], f'{i}')

        # Plot lines for neighboring points
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        len_colors = len(color_list)
        counter = 0        
        prev = 0            
        for i, j in sorted(self.neighbor_pairs):
            xi, yi, zi = final_coordinates[i]
            xj, yj, zj = final_coordinates[j]
            if i != prev:
                counter += 1
            ax.plot([xi, xj], [yi, yj], [zi, zj], '-', color=color_list[counter % len_colors])
            prev = j

        # Plot lines for specific distances
        if show_restraints:
            for i, j, d in self.desired_distances:
                xi, yi, zi = final_coordinates[i]
                xj, yj, zj = final_coordinates[j]
                ax.plot([xi, xj], [yi, yj], [zi, zj], 'g--', alpha=0.5, lw=1)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        return ax


def main():
    # Example data
    N = 10
    neighbor_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
    desired_distances = [(0, 2, 5.0), (1, 3, 7.0), (4, 6, 8.0), (5, 7, 6.0), (8, 9, 3.0)]

    # Number of iterations
    num_iterations = 100

    # Generate random initial guess
    initial_guess = np.random.rand(N * 3) * 10

    # Create an instance of SpatialOptimizer
    optimizer = SpatialOptimizer(N, neighbor_pairs, desired_distances)

    # Run the optimizations in parallel and get the solutions and scores
    solutions, scores = run_optimizations_parallel(optimizer, num_iterations, initial_guess)

    # Display the final coordinates for the first run as an example
    final_coordinates = solutions[0]
    for i, coord in enumerate(final_coordinates):
        print(f"Point {i}: {coord}")

    # Display the score for the first run
    print(f"Score for the first run: {scores[0]}")


if __name__ == "__main__":
    main()
