import numpy as np

def calc_ep():
    i = 35
    e = 1
    lr = 0.9975

    for i in range(0, i):
        e = lr * e

    print(e)

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


if __name__ == '__main__':
    from sklearn.preprocessing import normalize
    from ourgym import ActionMap

    am = ActionMap([-30, -15, -5, 0, 5, 15, 30])
