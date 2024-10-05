# Define the vector that contains the pauli opeations
import pandas as pd
import qibo
import numpy as np
from ansatz import build_hardware_efficient_ansatz
from model_params import LAMBDA_1, LAMBDA_2, LAMBDA_3, NLAYERS, NSHOTS, NUM_ASSETS, SIGMA_TARGET, TWO_QUBIT_GATES, K, N
from utils import string_to_int_list

# All this functions should help you build the cost function of the problem,
# which is the expected value of the Hamiltonian defined in (7).

# TODO hacer que el input sea A en vez de bit string


def A(i: int, bit_string: list[int]) -> float:
    """Building block of the hamiltonian. Note that we need to perform the
    change of variable x = (1-z)/2 where z are the eigenvalues of sigma_z.
    If we apply this change then this function depends on a bitstring,
    which is the outcome of quantum measurement. Make sure you undertand
    this point :)

    A_i on eq. 7
    Args:
        i (int): index of the asset to which it applies              
        bit_string (list[int]): bit string that encodes a portfolio.
        List with 0s and 1s.

    Returns:
        float: A_i, value for asset i.
    """
    # Constant multiplying the value
    const = np.array([2**(i-2) for i in range(K)], dtype=float)

    # Term under parenthesis in A_i. term|0> = 0, term|1> = 1
    my_vals = np.array([a for a in bit_string[i*K:(i+1)*K]], dtype=int)
    return np.dot(const, my_vals)


# Return term

def return_cost_function(dataset: pd.DataFrame, As: list[float])\
      -> float:
    """Corresponds to the first term of the expected value of the Hamiltonian
    in (7).

    Args:
        dataset (pd.DataFrame): dataset with the values obtained from Yahoo
            (X_i) for the daily logaritmic returns.
        bit_string (list[int]): bit string that encodes a portfolio.
            List with 0s and 1s.

    Returns:
        float: expected logaritmic return \sum_i A_i X_i, where here X_i
            is the average of the cost over all the analyzed days.
    """
    if (NUM_ASSETS != len(dataset.columns[0])):
        raise ValueError('Datased column number must coincide with model\
                         parameter NUM_ASSETS')
    Xs = np.zeros(NUM_ASSETS)  # Mean value of the X for each asset
    for i, col in enumerate(dataset.columns):
        Xs[i] = dataset[col].sum()/dataset[col].size

    return np.dot(As, Xs)


def risk_cost_function(dataset: pd.DataFrame,
                       As: list[float]) -> float:
    """Corresponds to the second term of the expected value of
    the Hamiltonian in (7).

    Args:
        dataset (pd.DataFrame):  dataset with the values obtained from Yahoo
            (X_i) for the daily logaritmic returns.
        bit_string (list[int]): bit string that encodes a portfolio.
            List with 0s and 1s.
    Returns:
        float: volatility (our volatility minus the target, squared)
    """
    sigma_ij = dataset.cov()
    expected_risk = np.dot(As, sigma_ij@As)
    return (expected_risk-SIGMA_TARGET**2)**2


def normalization_cost_function(As: list[float]) -> float:
    """Corresponds to the third term of the expected value of the Hamiltonian
    in (7).

    Args:
        dataset (pd.DataFrame):  dataset with the values obtained from Yahoo
            (X_i) for the daily logaritmic returns.
        bit_string (list[int]): bit string that encodes a portfolio.
            List with 0s and 1s.

    Returns:
        float: term in the hamiltonian ensuring the weights sum to 1.
    """

    return (np.sum(As)-1)**2


def compute_cost_function(dataset: pd.DataFrame, bit_string: list[int])\
      -> float:
    """Aggregates all the terms of the cost function.

    Args:
        dataset (pd.DataFrame):  dataset with the values obtained from Yahoo
            (X_i) for the daily logaritmic returns.
        bit_string (list[int]): bit string that encodes a portfolio.
            List with 0s and 1s.
    Returns:
        float: _description_
    """
    As = np.zeros(NUM_ASSETS)
    for i in range(NUM_ASSETS):
        # Calculate each A_i
        As[i] = A(i, bit_string)

    return -LAMBDA_1*return_cost_function(dataset, As) + \
        LAMBDA_2*risk_cost_function(dataset, As) + \
        LAMBDA_3*normalization_cost_function(As)
    
def compute_total_energy(parameters: list[float], circuit,
                         dataset: pd.DataFrame, nshots: int = NSHOTS,
                         num_qubits: int = N) -> float:
    """Aggregates the the energies of all the terms. This is the loss function
    and the parametrs are the ones optimized. First,
    use Circuit.set_parameters(parameters) to load the new set of parameters to
    the ansatz at every iteration of the optimization process. Second,
    measure the circuit and forward to result to energy functions. 

    Args:
        parameters (list[float]): _description_
        circuit (_type_): _description_
        dataset (pd.DataFrame): _description_
        nshots (_type_, optional): _description_. Defaults to NSHOTS.
        num_qubits (_type_, optional): _description_. Defaults to N.

    Returns:
        float: _description_
    """
    # Load parameters
    circuit.set_parameters(parameters) 
    # Get the frequencies of each bitstring
    # we get how many times each result is repeated
    result = circuit(nshots=nshots)
    results_freq = result.frequencies()  # By default is binary = True
    energy = 0
    for bit_sting in results_freq.keys():
        val = results_freq[bit_sting]
        energy += val*compute_cost_function(dataset, bit_sting)
    return energy