import os
import typing

import scipy
from sklearn import cluster
from sklearn.gaussian_process.kernels import *
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split

# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation

# Cost function constants
COST_W_UNDERPREDICT = 50.0
COST_W_NORMAL = 1.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict_pollution methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.train_x = None
        self.train_y = None
        self.km = None
        self.centers = None
        self.kd_tree = None
        self.gps = None
        self.k = 10
        self.rng = np.random.default_rng(seed=0)

        self.create_gp()

        # Given that we penalize underpredictions with w=50 in residential areas, this minimizes the expected loss
        # between our Gaussian processs f and our prediction p -> We choose the 93.1 percentile
        self.optimal_scale = 1.485
    
    def create_gp(self):
        """
        Create a Gaussian process for each local gp.
        """
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=0.2,
                                                           length_scale_bounds=(1e-2, 1e1), nu=1.5) \
                 + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e0))

        self.gps = [GaussianProcessRegressor(kernel=kernel,
                                             random_state=42, normalize_y=True, n_restarts_optimizer=1) for _ in
                    range(self.k)]

    
    
    def create_dataset(self, train_coordinates: np.ndarray, train_pollution_targets: np.ndarray):
        """
        Create a dataset for each local gp.
        """
        
        # Find centers for GPs
        self.find_centers(train_coordinates)

        labels = self.km.labels_  # (n,) cluster index per training point

        leaves = [np.where(labels == i)[0] for i in range(self.k)]  # list of index arrays

        # Analysis
        sizes = np.array([len(leaf) for leaf in leaves])  # or: np.bincount(labels, minlength=self.k)
        print(f"Points per cluster: {sizes.tolist()}")
        print(f"min={sizes.min()}, max={sizes.max()}, mean={sizes.mean():.1f}")

        self.train_x = [train_coordinates[idx] for idx in leaves]  # list of (n_i, d)
        self.train_y = [train_pollution_targets[idx] for idx in leaves]  # list of (n_i,)
    
    # Don't change the name or the signature of this function
    def fit_model(self, train_coordinates: np.ndarray, train_pollution_targets: np.ndarray, train_residential_flags: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_coordinates: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_pollution_targets: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        :param train_residential_flags: Binary variable denoting whether the 2D training point is in the residential area (1) or not (0)
        """
        self.create_dataset(train_coordinates, train_pollution_targets)

        for gp, X_train, y_train in zip(self.gps, self.train_x, self.train_y):
            gp.fit(X_train, y_train)

    # Don't change the name or the signature of this function
    def predict_pollution(self, test_coordinates: np.ndarray, test_residential_flags: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of city_areas.
        :param test_coordinates: city_areas as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param test_residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """

        "Find models for data point"
        index = self.km.predict(test_coordinates)

        order = np.argsort(index, kind='stable')  # test points sorted by GP
        bounds = np.cumsum(np.bincount(index, minlength=self.k))[:-1]
        groups = np.split(order, bounds)  # groups[i] = test indices for GP i

        gp_mean = np.empty(len(test_coordinates))
        gp_std = np.empty(len(test_coordinates))
        predictions = np.empty(len(test_coordinates))

        for gp, idx in zip(self.gps, groups):
            if len(idx) == 0:
                continue
            m, s = gp.predict(test_coordinates[idx], return_std=True)
            gp_mean[idx] = m
            gp_std[idx] = s ** 2
            # Todo Subtract std of noise
            predictions[idx] = m + self.optimal_scale*s*test_residential_flags[idx]

        return predictions, gp_mean, gp_std

    def find_centers(self, train_coordinates: np.ndarray):
        """
        Find the centers for the k local GPs
        """
        self.km = cluster.KMeans(
            n_clusters=self.k, init='k-means++', n_init='auto', max_iter=300,
            tol=1e-4, verbose=0, random_state=42, copy_x=True, algorithm='lloyd'
        ).fit(train_coordinates)

        self.centers = self.km.cluster_centers_



    def create_kdtree(self, train_coordinates):
        """
        Create a k-d tree for X_train. We want to find the nearest neighbors for each of the centers.
        """
        self.kd_tree= scipy.spatial.cKDTree(
            train_coordinates, leafsize=16, compact_nodes=True, copy_data=False,
            balanced_tree=True, boxsize=None
        )


# You don't have to change this function
def calculate_cost(ground_truth: np.ndarray, predictions: np.ndarray, residential_flags: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param ground_truth: Ground truth pollution levels as a 1d NumPy float array
    :param predictions: Predicted pollution levels as a 1d NumPy float array
    :param residential_flags: city_area info for every sample in a form of a bool array (NUM_SAMPLES,)
    :return: Total cost of all predictions as a single float
    """
    assert ground_truth.ndim == 1 and predictions.ndim == 1 and ground_truth.shape == predictions.shape

    # Unweighted cost
    cost = (ground_truth - predictions) ** 2
    weights = np.ones_like(cost) * COST_W_NORMAL

    # Case i): underprediction
    mask = (predictions < ground_truth) & [bool(residential_flag) for residential_flag in residential_flags]
    weights[mask] = COST_W_UNDERPREDICT

    # Weigh the cost and return the average
    return np.mean(cost * weights)


# You don't have to change this function
def is_inside_circle(coordinate, circle_parameters):
    """
    Checks if a coordinate is inside a circle.
    :param coordinate: 2D coordinate
    :param circle_parameters: 3D coordinate of the circle center and its radius
    :return: True if the coordinate is inside the circle, False otherwise
    """
    return (coordinate[0] - circle_parameters[0])**2 + (coordinate[1] - circle_parameters[1])**2 < circle_parameters[2]**2

# You don't have to change this function
def determine_residential_flags(grid_coordinates):
    """
    Determines the city_area index for each coordinate in the visualization grid.
    :param grid_coordinates: 2D coordinates of the visualization grid
    :return: 1D array of city_area indexes
    """
    # Circles coordinates
    circles = np.array([[0.5488135, 0.71518937, 0.17167342],
                    [0.79915856, 0.46147936, 0.1567626 ],
                    [0.26455561, 0.77423369, 0.10298338],
                    [0.6976312,  0.06022547, 0.04015634],
                    [0.31542835, 0.36371077, 0.17985623],
                    [0.15896958, 0.11037514, 0.07244247],
                    [0.82099323, 0.09710128, 0.08136552],
                    [0.41426299, 0.0641475,  0.04442035],
                    [0.09394051, 0.5759465,  0.08729856],
                    [0.84640867, 0.69947928, 0.04568374],
                    [0.23789282, 0.934214,   0.04039037],
                    [0.82076712, 0.90884372, 0.07434012],
                    [0.09961493, 0.94530153, 0.04755969],
                    [0.88172021, 0.2724369,  0.04483477],
                    [0.9425836,  0.6339977,  0.04979664]])

    residential_flags = np.zeros((grid_coordinates.shape[0],))

    for i,coordinate in enumerate(grid_coordinates):
        residential_flags[i] = any([is_inside_circle(coordinate, circ) for circ in circles])

    return residential_flags

# Don't change the name or the signature of this function
def perform_extended_model_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lon, grid_lat = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_grid = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    grid_residential_flags = determine_residential_flags(visualization_grid)

    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict_pollution(visualization_grid, grid_residential_flags)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0

    # Plot the actual predictions
    fig, ax = plt.subplots()
    ax.set_title('Extended visualization of task 1')
    im = ax.imshow(predictions, origin='lower', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()

# You do not need to change the name or the signature of this function; convenience helper used by main()
def get_city_area_data(train_x: np.ndarray, test_x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the city_area information from the training and test features.
    :param train_x: Training features
    :param test_x: Test features
    :return: Tuple of (training features' 2D coordinates, training features' city_area information,
        test features' 2D coordinates, test features' city_area information)
    """
    train_coordinates = np.zeros((train_x.shape[0], 2), dtype=float)
    train_residential_flags = np.zeros((train_x.shape[0],), dtype=bool)
    test_coordinates = np.zeros((test_x.shape[0], 2), dtype=float)
    test_residential_flags = np.zeros((test_x.shape[0],), dtype=bool)

    train_coordinates = train_x[:, :2]
    train_residential_flags = train_x[:, 2].astype(bool)
    test_coordinates = test_x[:, :2]
    test_residential_flags = test_x[:, 2].astype(bool)

    assert train_coordinates.shape[0] == train_residential_flags.shape[0] and test_coordinates.shape[0] == test_residential_flags.shape[0]
    assert train_coordinates.shape[1] == 2 and test_coordinates.shape[1] == 2
    assert train_residential_flags.ndim == 1 and test_residential_flags.ndim == 1

    return train_coordinates, train_residential_flags, test_coordinates, test_residential_flags

# you don't have to change this function
def main():
    # Load the training dataset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)
    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Extract the city_area information
    train_coordinates, train_residential_flags, test_coordinates, test_residential_flags = get_city_area_data(train_x, test_x)

    train_coordinates, val_coordinates, train_residential_flags, val_residential_flags, train_y, val_y \
        = train_test_split(
        train_coordinates,
        train_residential_flags,
        train_y,
        test_size=0.1,
        random_state=42,
    )

    # Fit the model
    print('Training model')
    model = Model()
    model.fit_model(train_coordinates, train_y, train_residential_flags)

    """print("kernel :", model.gpc.kernel_)
    # e.g. 0.8**2 * Matern(length_scale=[0.31, 0.29], nu=1.5) + 0.5**2 * Matern(length_scale=[0.03, 0.03], nu=1.5) + WhiteKernel(noise_level=2.1)
    print("log_marginal_likelihood:", model.gpc.log_marginal_likelihood_value_)"""

    # Predict on the test features
    print('Predicting on test features')
    predictions, gp_mean, gp_std = model.predict_pollution(val_coordinates, val_residential_flags)
    print(predictions)
    # loss calculation
    cost = calculate_cost(val_y, predictions, val_residential_flags)
    print(f"Validation cost: {cost}")

    if EXTENDED_EVALUATION:
        perform_extended_model_evaluation(model, output_dir='.')


if __name__ == "__main__":
    main()