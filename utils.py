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



#functionalized version of code presented in the MCMC tutorial
def spins(steps=STEPS, random=True, temp=T, size=N, J=J, KB=KB):
    """
    Parameters:
    - steps (int): The number of Monte Carlo steps to perform (default: STEPS).
    - random (bool): If True, initialize the lattice with random spins. If False, initialize with all spins up (default: True).
    - temp (float): The temperature of the system (default: T).
    - size (int): The size of the lattice (default: N).
    - J (float): The coupling constant (default: J).
    - KB (float): The Boltzmann constant (default: KB).
    """
  
    #the mask will be used to precompute the neghbors of each point in the lattice 
    mask = np.zeros((N,N,N,N), dtype=np.byte) # create a NxN mask for each coordinate pair (i,j) i<N, j<N
    for i in range(N):  #populate the mask with the neghbors at each position
        for j in range(N):
            mask[i,j,i-1,j] = 1 # underflow is automatic because numpy understands [-1] to be the far end of row or col
            mask[i,j,i,j-1] = 1
            mask[i,j,i+1 if i<N-1 else 0,j] = 1 # account for overflow
            mask[i,j,i,j+1 if j<N-1 else 0] = 1
    #the over/underflow condations may be reffered to as boundary conditions 

    if random:
        lattice_spins = random_spins(size)
    else:
        lattice_spins = np.ones((size,size), dtype=np.byte)

    num_accept = 0
    m_values = []
    for t in tqdm(range(steps)):
        i, j = np.random.randint(size), np.random.randint(size)
        # we only need to consider the neighbors of
        # (i, j) to calculate the change in energy
        delta_energy = np.sum(-J * -2 * lattice_spins[i, j] * mask[i, j]) # removed inner loop
        if delta_energy <= 0:
            lattice_spins[i, j] *= -1
            num_accept += 1
        elif delta_energy > 0:
            prob = np.exp(-delta_energy / (KB * temp))
        if np.random.random() < prob:
            lattice_spins[i, j] *= -1
            num_accept += 1
        m_values.append(np.mean(lattice_spins))
    return m_values, lattice_spins, steps

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

def stats(temp, steps=STEPS, burn_in=BURNIN, size=N):

    m_values, _, __ = spins(steps, random=False, temp=T, size=N)
    return np.mean(m_values[burn_in:]), np.std(m_values[burn_in:])
