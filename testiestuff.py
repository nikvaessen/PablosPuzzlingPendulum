import numpy as np

def create_Q(state_dim, action_dim, low, high, delta):
    if len(low) != state_dim or len(high) != state_dim or len(delta) != state_dim:
        raise ValueError("length of low, high and delta arrays do not match state dimension")

    max_indexes = np.zeros((len(delta) + 1), dtype=np.int32)
    max_indexes[:-1] = delta
    max_indexes[-1] = action_dim

    num_entries = np.array(max_indexes).prod()
    Q = np.zeros((num_entries, state_dim + 2))  # one for q values, one for action

    options = [list() for _ in low]
    for idx, l in enumerate(options):
        for bound in np.linspace(low[idx], high[idx], delta[idx]):
            l.append(np.round(bound, 2))

    options.append(np.array([a for a in range(0, action_dim)], dtype=np.int32))

    indexes = [0 for _ in range(0, state_dim + 1)]
    row = 0
    while row < num_entries:
        Q[row, 0:-1] = np.array([options[i][j] for i, j in enumerate(indexes)])

        row += 1

        idx_to_update = state_dim
        while idx_to_update > -1:
            indexes[idx_to_update] += 1

            if indexes[idx_to_update] >= max_indexes[idx_to_update]:
                indexes[idx_to_update] = 0
                idx_to_update -= 1
            else:
                break

    return Q


def test():

    # Q = np.array([[0, 0, 0, 0],
    #               [0, 0, 1, 0],
    #               [0, 1, 0, 0],
    #               [0, 1, 1, 0],
    #               [0, 2, 0, 0],
    #               [0, 2, 1, 0],
    #               [1, 0, 0, 0],
    #               [1, 0, 1, 0],
    #               [1, 1, 0, 0],
    #               [1, 1, 1, 0],
    #               [1, 2, 0, 0],
    #               [1, 2, 1, 0],
    #               [2, 0, 0, 0],
    #               [2, 0, 1, 0],
    #               [2, 1, 0, 0],
    #               [2, 1, 1, 0],
    #               [2, 2, 0, 0],
    #               [2, 2, 1, 0]])

    state_dim = 2
    action_dim = 2

    low = np.array([0, 0])
    high = np.array([5, 5])
    delta = np.array([5, 5])

    Q = create_Q(state_dim, action_dim, low, high, delta)

    for row in range(0, Q.shape[0]):
        print(row, Q[row, :])


    s = [5, 3]

    print(Q[:, 0:2])

    valid = (Q[:, 0:2] <= s).all(axis=1)
    indexes = np.where(valid == True)[0][-action_dim:]

    print(valid)


    print(len(indexes))
    print(indexes)
    print(indexes[0])


if __name__ == '__main__':
    print(np.random.rand())

    a = np.array([1, 2, 3, 4])

    for _ in range(100):
        print(np.random.choice(a))


    # t1 = np.array([1, 1, 1, 1])
    # t2 = np.array([0, 0, 0])
    # t3 = np.array([10, 10, 10, 10])
    # t4 = np.array([0, 0, 10])
    # t5 = np.array([1, 1, 1, 5])
    #
    # print(t1.all())
    # print(t2.all())
    # print(t3.all())
    # print(t4.all())
    # print(t5.all())

