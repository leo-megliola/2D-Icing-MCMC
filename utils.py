import numpy as np 
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import os
import spins_module

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

    spins = np.random.randint(2, size=(n, n), dtype=np.int32)  # random 0 or 1
    return np.where(spins, 1, -1).astype(np.int32)  # the zeroes will evaluate to false and become -1

def spin_test():
    N = 1_000_000
    lattice = np.ones((100,100), dtype=np.int32)
    #lattice = random_spins(100)
    m = np.zeros((N), dtype=np.float64)
    spins_module.spins(lattice, m, N, 0.0, 2.0, 1, 1)
    return m

def spins(steps = STEPS, random = True, temp = T, size=N, J=J, KB=KB, B=B, display=True):
    
    if random:
        lattice_spins = random_spins(size)
    else:
        lattice_spins = np.ones((size,size), dtype=np.int32)
    m_values = np.zeros((steps), np.float64)              #init in py, then pass to cpp
    
    spins_module.spins(lattice_spins, m_values, steps, B, temp, J, KB) #call cpp to preform MCMC
    plt.plot(m_values)
    return m_values, lattice_spins, steps

def spins_py(steps=STEPS, random=True, temp=T, size=N, J=J, KB=KB, B=B, display=True):
    """
    Simulates the 2D Ising model using the Metropolis Monte Carlo algorithm.

    Parameters:
    - steps (int): Number of Monte Carlo steps to perform (default: STEPS).
    - random (bool): If True, initialize the lattice with random spins. If False, initialize with all spins up (default: True).
    - temp (float): Temperature of the system (default: T).
    - size (int): Size of the lattice (default: N).
    - J (float): Coupling constant (default: J).
    - KB (float): Boltzmann constant (default: KB).
    - B (float): External magnetic field (default: B).
    - display (bool): If True, display a progress bar during simulation (default: True).

    Returns:
    - m_values (ndarray): Array of magnetization values at each step.
    - lattice_spins (ndarray): Final configuration of spins on the lattice.
    - steps (int): Number of Monte Carlo steps performed.
    """
    
    if random:
        lattice_spins = random_spins(size)
    else:
        lattice_spins = np.ones((size, size))

    total = np.sum(lattice_spins)
    m_values = np.zeros((steps))

    num_accept = 0
    for t in tqdm(range(steps), disable=not display):
        i, j = np.random.randint(size), np.random.randint(size)
        delta_energy = 0
        for k, l in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            i_neigh = i + k if i + k < N else 0
            j_neigh = j + l if j + l < N else 0
            delta_energy += -J * -2 * lattice_spins[i, j] * lattice_spins[i_neigh, j_neigh]
        delta_energy += -B * lattice_spins[i, j]
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
    return m_values / (size * size), lattice_spins, steps



def plot_lattice(lattice_spins, N=N, savefig=None, show_plot=True):
    """
    Plot the lattice spins and optionally save the figure with a transparent background.

    Parameters:
    - lattice_spins (numpy.ndarray): The array representing the lattice spins.
    - N (int): The size of the lattice.
    - savefig (str): Path to save the figure.
    - show_plot (bool): Whether to show the plot.
    """
    if not os.path.exists('figs') and savefig is not None:
        os.makedirs('figs')

    plt.figure()
    plt.imshow(lattice_spins, cmap="RdYlBu")
    for i in range(N):
        plt.axhline(i + 0.5, color="black", lw=0.1)
        plt.axvline(i + 0.5, color="black", lw=0.1)
    
    plt.axis('off')  # This will remove the axes
    
    if show_plot:
        plt.show()

    if savefig is not None:
        # Save the figure with a transparent background
        plt.savefig(os.path.join('figs', savefig), transparent=True)
        plt.close()





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

    m_values, _, __ = spins(steps, random=False, temp=temp, size=size, B=B, display=display)
    return np.mean(m_values[burn_in:]), np.std(m_values[burn_in:])

import imageio

def spins_with_snapshots(steps=STEPS, random=True, temp=T, size=N, J=J, KB=KB, B=B, display=True, snapshot_interval=1000):
    """
    This function is an adaptation of the spins function, which also captures snapshots
    of the lattice at each step for later visualization in a GIF.

    Parameters are similar to the spins function with an added snapshot_interval parameter
    that determines how frequently to capture the lattice state.
    
    Returns a list of lattice configurations at each step.
    """
    lattice_configurations = []
    lattice_spins = random_spins(size) if random else np.ones((size, size))

    total = np.sum(lattice_spins)
    m_values = np.zeros((steps))

    num_accept = 0
    for t in tqdm(range(steps), disable=not display):
        i, j = np.random.randint(size), np.random.randint(size)
        delta_energy = 0
        for k, l in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            i_neigh = i + k if i + k < N else 0
            j_neigh = j + l if j + l < N else 0
            delta_energy += -J * -2 * lattice_spins[i, j] * lattice_spins[i_neigh, j_neigh]
        delta_energy += -B * lattice_spins[i, j]
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
        if t % snapshot_interval == 0:
            lattice_configurations.append(np.copy(lattice_spins))

    return lattice_configurations

def generate_ising_gif(lattice_configurations, N=N, gif_name='ising_simulation.gif', duration=0.1):
    """
    Generates and saves a GIF from the lattice configurations.

    Parameters:
    - lattice_configurations (list): List of 2D arrays representing lattice states over time.
    - N (int): The size of the lattice.
    - gif_name (str): Name of the output GIF file.
    - duration (float): Duration of each frame in the GIF.
    """
    image_filenames = []
    for index, lattice in tqdm(enumerate(lattice_configurations), total=len(lattice_configurations), desc='Creating GIF'):
        filename = f'temp_lattice_{index}.png'
        actual_filename = os.path.join('figs', filename)
        plot_lattice(lattice, N, savefig=filename, show_plot=False)
        image_filenames.append(actual_filename)

    print(image_filenames)
    with imageio.get_writer(gif_name, mode='I', duration=duration) as writer:
        for filename in image_filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        
    # Clean up temporary image files
    for filename in image_filenames:
        os.remove(filename)
