import matplotlib.pyplot as plt
import numpy as np

def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    # use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def train_2d(trainer, steps=20):
    x1, x2 = -5, -2
    results = [(x1, x2)]
    for i in range(steps):
        x1, x2,_,_ = trainer(x1, x2)
        results.append((x1, x2))
    print('epoch %d, x1 %f, x2 %f' % (i + 1, x1, x2))
    return results

def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-5.5, 1.0, 0.1), np.arange(-3.0, 1.0, 0.1))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

#Momentum
import tensorflow as tf
eta = 0.4

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

def gd_2d(x1, x2):
    return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2,0,0)

show_trace_2d(f_2d, train_2d(gd_2d))

eta, beta = 0.4, 0.5
def momentum_2d(x1, x2, v1, v2):
    v1 = beta * v1 + eta * 0.2 * x1
    v2 = beta * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

show_trace_2d(f_2d, train_2d(momentum_2d))

