from concrete.fhe import Compiler, round_bit_pattern, truncate_bit_pattern
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return (x>0) * x


def f(x, bias:float=2.3, scale=.8):
    return relu(x.astype(np.float64)*scale + bias).astype(np.int64)


def main():
    x = np.arange(-23, 52)
    y = f(x)
    fig, ax = plt.s
    return

if __name__ == "__main__":
    main()
