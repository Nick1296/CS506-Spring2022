import numpy as np


def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i]) ** 2
    return res ** (1 / 2)


def manhattan_dist(x, y):
    if len(x) != len(y):
        raise ValueError("lengths must be equal")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("lengths must not be zero")

    x_arr = np.array(x)
    y_arr = np.array(y)
    return np.sum(np.abs(x_arr - y_arr))


def jaccard_dist(x, y):
    if len(x) == 0 or len(y) == 0:
        raise ValueError("lengths must not be zero")
    inter = []
    union = []
    for x_elem in x:
        if x_elem not in union:
            union.append(x_elem)
            if x_elem in y:
                inter.append(x_elem)
    for y_elem in y:
        if y_elem not in union:
            union.append(y_elem)
    print(f"inter:{inter}, union:{union}")
    if len(union) == 0:
        raise ValueError("empty union!")
    else:
        return 1 - (len(inter) / len(union))


def cosine_sim(x, y):
    if len(x) != len(y):
        raise ValueError("lengths must be equal")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("lengths must not be zero")
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


# Feel free to add more
