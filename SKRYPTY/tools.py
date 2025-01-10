
import numpy as np
import matplotlib.pyplot as plt

class myFilter:

    def __init__(self, dt = 1.0, tau=10.0, state = 0.0):
        self.state = state
        self.tau = tau
        self.dt = dt

    def set_state(self, state):
        self.state = state

    def low_pass_filter(self, input):
        alfa = self.dt / (self.tau + self.dt)
        self.state = input * alfa + self.state * (1 - alfa)
        return self.state

    def high_pass_filter(self, input):
        self.low_pass_filter(input)
        return input - self.state

    def get_state(self):
        return self.state


def main():
    filterLow = myFilter(dt=1.0, tau = 10.0, state = 0.0)
    filterHigh = myFilter(dt=1.0, tau = 10.0, state = 0.0)

    input = np.ones(50)
    filteredLow = np.zeros(50)
    filteredHigh = np.zeros(50)

    for i in range(len(input)):
        inp = input[i]
        filteredLow[i] = filterLow.low_pass_filter(inp)
        filteredHigh[i] = filterHigh.high_pass_filter(inp)

    plt.plot(input)
    plt.plot(filteredLow)
    plt.plot(filteredHigh)
    plt.show()

if __name__ == "__main__":
    main()
