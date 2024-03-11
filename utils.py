import numpy as np 
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os


# utils.py
import yaml

# Function to load variables from the YAML configuration file
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load the common variables
common_variables = load_config()

# Now, you can use these variables within your utils.py or export them
STEPS = common_variables['STEPS']
N = common_variables['N']
J = common_variables['J']
KB = common_variables['KB']
T = common_variables['T']
BURNIN = common_variables['BURNIN']
B = common_variables['B']


def random_spins(n):
    """
    Generate a random spin configuration of size n x n.

    Parameters:
    n (int): The size of the spin configuration.

    Returns:
    numpy.ndarray: A 2D array representing the random spin configuration, where 1 represents spin up and -1 represents spin down.
    """

    spins = np.random.randint(2, size=(n, n))  # random 0 or 1
    return np.where(spins, 1, -1)  # the zeroes will evaluate to false and become -1


def spins(steps = STEPS, random = True, temp = T, size=N, J=J, KB=KB, B=B, display=True):
    if random:
        lattice_spins = random_spins(size)
    else:
        lattice_spins = np.ones((size,size))

    total = np.sum(lattice_spins)
    m_values = np.zeros((steps))

    num_accept = 0
    for t in tqdm(range(steps)):
        i, j = np.random.randint(size), np.random.randint(size)
        delta_energy = 0
        for k, l in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            i_neigh = i + k if i + k < N else 0
            j_neigh = j + l if j + l < N else 0
            #delta_energy = np.sum(-J * -2 * lattice_spins[i, j] * mask[i, j]) - B * lattice_spins[i, j] <-- Dima implementation in old code pattern 
            delta_energy += -J * -2 * lattice_spins[i, j] * lattice_spins[i_neigh, j_neigh] #'-2' UNREMOVED
        delta_energy += - B * lattice_spins[i, j] #moved outside of the for loop; only acts once on the lattice point in question (i,j)
        if delta_energy <= 0:
            total -= lattice_spins[i, j]
            lattice_spins[i, j] *= -1
            total += lattice_spins[i, j]
            num_accept += 1
        else:
            prob = np.exp(-delta_energy / (KB * temp))
            if np.random.random() < prob:
                total -= lattice_spins[i, j]
                lattice_spins[i, j] *= -1
                total += lattice_spins[i, j]
                num_accept += 1
        m_values[t] = total
    return m_values/(size*size), lattice_spins, steps


def plot_lattice(lattice_spins, N=N, savefig = None):
    """
    Plot the lattice spins.

    Parameters:
    lattice_spins (numpy.ndarray): The array representing the lattice spins.
    N (int): The size of the lattice.

    Returns:
    None
    """

    plt.figure()
    plt.imshow(lattice_spins, cmap="RdYlBu")
    # show gridlines
    for i in range(N):
        plt.axhline(i + 0.5, color="black", lw=0.1)
        plt.axvline(i + 0.5, color="black", lw=0.1)
    plt.show()
    if savefig is not None:
        plt.savefig(os.path.join('figs', savefig))   




def plot_m(m_values, burn_in=BURNIN, temp=T, savefig = None):
    """
    Plots the magnetization values over steps and displays statistics.

    Parameters:
    - m_values (array-like): Array of magnetization values.
    - burn_in (int): Number of burn-in steps to exclude from the plot (default: BURNIN).
    - temp (float): Temperature value (default: T).

    Returns:
    None
    """
    plt.style.use("ggplot")

    m_mean = np.mean(m_values[burn_in:])
    m_std = np.std(m_values[burn_in:])

    plt.figure()
    plt.plot(range(STEPS)[:burn_in], m_values[:burn_in], label="Burn in")
    plt.plot(range(STEPS)[burn_in:], m_values[burn_in:], label="Sampling")
    # show mean as dashed line
    plt.plot(range(STEPS), m_mean * np.ones((STEPS)), "--", color="black")
    # show variance as filled box
    plt.fill_between(range(STEPS), m_mean - m_std, m_mean + m_std, color="gray", alpha=0.3)
    plt.xlabel("Steps")
    plt.ylabel("Magnetization")
    plt.legend(title=f"T={temp}")
    plt.title(f"Burn in = {burn_in}; T = {temp}")
    plt.show()
    if savefig is not None:
        plt.savefig(os.path.join('figs', savefig))

    print(f"magnetization mean = {m_mean}")
    print(f"magnetization std = {m_std}")

def stats(temp=T, steps=STEPS, burn_in=BURNIN, size=N, B=B, display=True):

    m_values, _, __ = spins(steps, random=False, temp=temp, size=size, B=B)
    return np.mean(m_values[burn_in:]), np.std(m_values[burn_in:])