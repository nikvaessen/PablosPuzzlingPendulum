from numpy import sin, cos, arange, pi

interval = 0.05 * 2 * pi
midpoints = arange(0, 2 * pi, interval)


def linearise_sin(val):
    closest_multiple = int(value / (2 * pi))
    range_adjusted_val = val - closest_multiple * 2 * pi + (2 * pi if val < 0 else 0)
    midpoint = min(midpoints, key=lambda x: abs(x - range_adjusted_val))
    midpoint_value = sin(midpoint)
    derivative = cos(midpoint)
    return midpoint_value + derivative * (range_adjusted_val - midpoint)


def linearise_cos(val):
    closest_multiple = int(value / (2 * pi))
    range_adjusted_val = val - closest_multiple * 2 * pi + (2 * pi if val < 0 else 0)
    midpoint = min(midpoints, key=lambda x: abs(x - range_adjusted_val))
    midpoint_value = cos(midpoint)
    derivative = -sin(midpoint)
    return midpoint_value + derivative * (range_adjusted_val - midpoint)


if __name__ == '__main__':
    while True:
        value = float(input('Please enter a value: '))
        print(linearise_sin(value), sin(value))
        print(linearise_cos(value), cos(value))