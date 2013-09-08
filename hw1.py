import numpy as np
import matplotlib.pyplot as plt

def hwplotprep():
    plt.grid()
    gcf = plt.gcf()
    gcf.set_size_inches(15 / 2.54, 10 / 2.54)
    gcf.set_tight_layout(True)
    pass

def main():
    z = np.arange(-5, 5, 0.1)
    #fz = 1/1
    fz = 1. / (1 + np.exp(-z))
    plt.plot(z, fz, linewidth=3)
    plt.xlabel('z')
    plt.ylabel('f(z)')  # we always label axes, yes we do!
    hwplotprep()
    plt.savefig('sigmoid.pdf')

if __name__ == '__main__':
    main()