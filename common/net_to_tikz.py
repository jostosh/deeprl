import numpy as np

def conv_drawing(conv):

    x0 = conv['x0'] - conv['size'] / 2.
    y0 = conv['y0'] - conv['size'] / 2. + conv['n_filters'] / 20.

    return '\n'.join(["\draw [very thin, draw=black, fill={color}, fill opacity=0.25] "
                      "({x0:.2f}, {y0:.2f}) -- ++({size}, 0) -- ++(0, {size}) -- ++(-{size}, 0) -- cycle;"
                     .format(color=conv['color'], x0=x0 + i / 10., y0=y0 - i / 10., size=conv['size'])
                      for i in range(conv['n_filters'])]) + '\n'


def get_x0(layer):
    return layer['x0'] - layer['size'] / 2.


def get_y0(layer):
    return layer['y0'] - layer['size'] / 2.


def connect_convs(conv1, conv2):
    ret = ""

    offset_x = np.random.rand()
    offset_y = np.random.rand()

    def patch(conv, ret, forced_size=None):
        x0 = get_x0(conv) + (conv['n_filters'] - 1) / 10.
        y0 = get_y0(conv) - (conv['n_filters'] - 1) / 20.

        patch_size = conv['size'] / 4. if not forced_size else forced_size
        x_patch = x0 + offset_x * (conv['size'] - patch_size)
        y_patch = y0 + offset_y * (conv['size'] - patch_size)

        ret += "\draw [very thin, draw=black, fill={color}!50!black!50, fill opacity=0.5] " \
               "({x0:.2f}, {y0:.2f}) -- ++({size}, 0) -- ++(0, {size}) -- ++(-{size}, 0) -- cycle;\n".format(x0=x_patch,
                                                                                                             y0=y_patch,
                                                                                                             color=conv['color'],
                                                                                                             size=patch_size)
        return ret, x_patch, y_patch, patch_size

    ret, x_patch1, y_patch1, patch_size1 = patch(conv1, ret)
    ret, x_patch2, y_patch2, patch_size2 = patch(conv2, ret, 0.05)

    for x0, y0, x1, y1 in zip([x_patch1, x_patch1 + patch_size1, x_patch1, x_patch1 + patch_size1],
                              [y_patch1, y_patch1 + patch_size1, y_patch1 + patch_size1, y_patch1],
                              [x_patch2, x_patch2 + patch_size2, x_patch2, x_patch2 + patch_size2],
                              [y_patch2, y_patch2 + patch_size2, y_patch2 + patch_size2, y_patch2]):
        ret += "\draw [ultra thin; draw=black] ({0:.2f},{1:.2f}) -- ({2:.2f},{3:.2f});\n".format(x0, y0, x1, y1)

    return ret

def fc_drawing(fc):
    circle_size = fc['height'] / (3. * fc['n_neurons'])
    x0 = fc['x0']
    y0 = fc['y0'] + fc['height'] / 2.

    ret = ""

    for y0_ in np.linspace(y0, y0 - fc['height'], fc['n_neurons']):
        ret += "\draw [ultra thin, draw=black, fill={color}] ({x0:.2f}, {y0_:.2f}) circle ({size});\n"\
            .format(color=fc['color'],
                    x0=x0,
                    y0_=y0_,
                    size=circle_size)

    return ret


if __name__ == "__main__":
    conv1 = {
        'size': 2,
        'n_filters': 8,
        'color': 'red',
        'x0': 0,
        'y0': 0
    }
    conv2 = {
        'size': 1,
        'n_filters': 12,
        'color': 'blue',
        'x0': 2.5,
        'y0': 0
    }
    fc = {
        'height': 2,
        'n_neurons': 8,
        'x0': 5,
        'y0': 0,
        'color': 'yellow'
    }

    with open('example.tex', 'w') as f:
        f.write(conv_drawing(conv1))
        f.write(conv_drawing(conv2))
        f.write(connect_convs(conv1, conv2))
        f.write(fc_drawing(fc))
