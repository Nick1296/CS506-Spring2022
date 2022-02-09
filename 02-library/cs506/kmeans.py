from collections import defaultdict
from math import inf
import random
from cs506 import sim


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)

    Returns a new point which is the center of all the points.
    """
    x = 0
    y = 0
    for elem in points:
        x += elem[0]
        y += elem[1]
    mean = [x / len(points), y / len(points)]
    return mean


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    assignments_dict = {}
    centers = []
    for i in range(0, len(assignments)):
        points = assignments_dict.get(assignments[i])
        if points is None:
            points = []
        points.append(dataset[i])
        assignments_dict.update({assignments[i]: points})
    for p_list in sorted(assignments_dict.keys()):
        centers.append(point_avg(assignments_dict[p_list]))
    return centers


def assign_points(data_points, centers):
    """ """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    return sim.euclidean_dist(a, b)


def distance_squared(a, b):
    return distance(a, b) ** 2


def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    points = []
    for i in range(0, k):
        rnd = random.randint(0, len(dataset) - 1)
        points.append(dataset[rnd])
    return points


def cost_function(clustering):
    cost = 0
    for center_id in clustering:
        center = point_avg(clustering[center_id])
        for point in clustering[center_id]:
            cost += distance(center, point)
    return cost


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    centers = [dataset[random.randint(0, len(dataset) - 1)]]
    for i in range(0, k - 1):
        distances = {}
        total_distance = 0
        for point in dataset:
            tmp = 0
            for center in centers:
                tmp += distance_squared(center, point)
            total_distance += tmp
            distances.update({tmp: point})
        num = random.randint(0, total_distance)
        sorted_dist = sorted(distances.keys())
        cumul_dist = 0
        for dist in sorted_dist:
            cumul_dist += dist
            if num <= cumul_dist:
                centers.append(distances[dist])
                break
    return centers


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset) + 1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset) + 1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
