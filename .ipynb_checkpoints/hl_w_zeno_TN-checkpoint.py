import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from itertools import product
from utilities import *
import numpy as np
from tenpy.models.lattice import Chain
from tenpy.networks.site import SpinHalfSite
from tenpy.models.model import MPOModel
from tenpy.algorithms import tebd
from tenpy.networks.mps import MPS
from tenpy.models.model import NearestNeighborModel, CouplingModel
from tenpy.linalg import np_conserved as npc

##############################################################################################################
# Utility functions
###############################################################################################################

def print_matrix(matrix, precision=1e-10):
    """
    Pretty print matrix function that prints just non-zero elements, up to a certain precision.
    If 0 prints 0 and not 1e-16 like things, moreover put a nice formatting display.
    If the imaginary part is 0, it does not print it.
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.abs(matrix[i,j]) > precision:
                if np.abs(np.imag(matrix[i,j])) < precision:
                    print("{:5.2f}".format(np.real(matrix[i,j])), end=" ")
                elif np.abs(np.real(matrix[i,j])) < precision:
                    print("{:5.2f}j".format(np.imag(matrix[i,j])), end=" ")
                else:
                    print("{:5.2f}+{:5.2f}j".format(np.real(matrix[i,j]), np.imag(matrix[i,j])), end=" ")
            else:
                print("{:5.2f}".format(0), end=" ")
        print()

##############################################################################################################
# Operate with tensors
###############################################################################################################

def LocalProduct(Psi, Operators):
    """
    Calculate the product (A1xA2x...xAn)|psi>
    """
    for i, op in enumerate(Operators):
        if op != 'Id':
            Psi.apply_local_op(i, op)
    return Psi

###############################################################################################################
# Zeno protocol and tomography essentials
###############################################################################################################

def generate_U_kick_from_list(psi_0_list):
    """
    Generate the kick operator from the full state structure.
    """
    # put a Z in correspondance of the "0" and identity otherwise
    kick_dict = {'0': "Sigmaz", 'F': "Id", 'L': "Id"}
    # create the kick operator list
    U_kick = []
    for psi_0 in psi_0_list:
        U_kick.append(kick_dict.get(psi_0, "Id"))
        if psi_0 == 'F':
            U_kick.append("Id")
    return U_kick


def get_qpt_keys(n, complete_set=False, exte=False):
    """
    Generate the measurement keys for quantum process tomography. This 
    is meant for 2 qubits. 
    """
    sq_states = []
    # make single qubit states, then tensor them 
    psi_0 = 'Z0'
    sq_states.append(psi_0)
    psi_1 = 'Z1'
    sq_states.append(psi_1)
    psi_plus = 'X0'
    sq_states.append(psi_plus)
    if complete_set:
        psi_minus = 'X1'
        sq_states.append(psi_minus)
    psi_L = 'Y0'
    sq_states.append(psi_L)
    if complete_set:
        psi_R = 'Y1'
        sq_states.append(psi_R)

    # tensor them
    qpt_keys = []
    for state_combination in product(sq_states, repeat=n):
        psi_combined = state_combination[0]
        for psi in state_combination[1:]:
            psi_combined = psi_combined + psi
        qpt_keys.append(psi_combined)

    if not exte:
        # remove 0 and 1 subscript 
        qpt_keys = [key.replace('0', '').replace('1', '') for key in qpt_keys]
        # remove double elements, don't change the order
        qpt_keys = list(dict.fromkeys(qpt_keys))
    return qpt_keys


site = SpinHalfSite(conserve=None)
Had = (site.get_op('Sigmax') + site.get_op('Sigmaz'))/ np.sqrt(2)
S_dag = (1+1.j)*0.5*site.get_op('Sigmaz')+(1-1.j)*0.5*site.get_op('Id')
H_Sdag = npc.tensordot(Had, S_dag, axes=[['p*'], ['p']])

measurement_rotations = {
    'Z': "Id",
    'X': Had,  # X basis
    'Y': H_Sdag,  # Y basis
    'I': "Id"  # Identity (no rotation)
}

def get_probabilities(full_state_list, psi, meas_keys, N_shots=None):
    """ 
    Calculate the probabilities of measuring the given states in the specified measurement bases.
    """
    basis_keys = [key.replace('0', '').replace('1', '') for key in meas_keys]
    basis_keys = list(dict.fromkeys(basis_keys))
    probs = {}
    qubit_indices = []
    counter = 0
    for key in full_state_list:
        if key == 'F':
            qubit_indices.append(str(counter)+str(counter+1))
            probs[str(counter)+str(counter+1)] = {}
            counter += 2
        else:
            counter += 1
    for key in basis_keys:
        meas_list = ['I']*len(full_state_list)
        meas_list = [op if full_state_list[i] != 'F' else key for i, op in enumerate(meas_list)]
        meas_list = [item for sublist in meas_list for item in (sublist if len(sublist)>1 else [sublist])]
        # create rotation matrices for each measurement basis
        state_rotations = [measurement_rotations[op] for op in meas_list]
        # rotate state depending on the measurement basis
        state = LocalProduct(psi.copy(), state_rotations)
        for qubits in qubit_indices:
            mid = len(qubits) // 2
            q0 = int(qubits[:mid])
            q1 = int(qubits[mid:])
            # get reduced rho
            rho_q0q1 = state.get_rho_segment([q0, q1])
            rho_q0q1 = rho_q0q1.to_ndarray().reshape(4, 4)
            p00 = rho_q0q1[0, 0].real
            p01 = rho_q0q1[1, 1].real
            p10 = rho_q0q1[2, 2].real
            p11 = rho_q0q1[3, 3].real
            if N_shots is not None:
                # simulate N_shots measurements
                p00 = np.random.binomial(N_shots, p00) / N_shots
                p01 = np.random.binomial(N_shots, p01) / N_shots
                p10 = np.random.binomial(N_shots, p10) / N_shots
                p11 = np.random.binomial(N_shots, p11) / N_shots
            probs[qubits][key[0]+'0'+key[1]+'0'] = p00
            probs[qubits][key[0]+'0'+key[1]+'1'] = p01
            probs[qubits][key[0]+'1'+key[1]+'0'] = p10
            probs[qubits][key[0]+'1'+key[1]+'1'] = p11
    return probs


def generate_psi0_list(N, reshaping_step):
    """
    Generate the list of initial states for the given reshaping step.
    """
    psi_0_list = []
    flag = True if reshaping_step == 0 or reshaping_step == 2 else False
    i = 0
    while i < N:
        if flag:
            if reshaping_step == 2 and i == 0:
                psi_0_list.append('L')
                i+=1
                continue
            psi_0_list.append('0')
            i+=1
            flag = False
        elif N-i == 1:
            psi_0_list.append('L')
            i+=1
        else:
            psi_0_list.append('F')
            flag = True
            i+=2
    return psi_0_list


def get_fiducial_states(n, complete_set=False):
    """
    Generate fiducial product states for tomography
    """
    sq_states = []
    # Define single-qubit states as lists of amplitudes
    psi_0 = [1, 0]
    sq_states.append(psi_0)
    psi_1 = [0, 1]
    sq_states.append(psi_1)
    psi_plus = [1/np.sqrt(2), 1/np.sqrt(2)]
    sq_states.append(psi_plus)
    if complete_set:
        psi_minus = [1/np.sqrt(2), -1/np.sqrt(2)]
        sq_states.append(psi_minus)
    psi_L = [1/np.sqrt(2), 1.j/np.sqrt(2)]
    sq_states.append(psi_L)
    if complete_set:
        psi_R = [1/np.sqrt(2), -1.j/np.sqrt(2)]
        sq_states.append(psi_R)

    # Build all product states as MPS
    fiducial_states = []
    for state_combination in product(sq_states, repeat=n):
        # Each site gets its local state as a numpy array
        local_states = [np.array(s, dtype=complex) for s in state_combination]
        # Create the MPS from local product states
        fiducial_states.append(local_states)
    return fiducial_states


def generate_psi0_fromlist(psi_0_list, fiducial_state, N):
    """
    Generate the initial state from the given list of states.
    """
    states_dict = {'F': fiducial_state, 'L': np.array([1/np.sqrt(2), 1.j/np.sqrt(2)], dtype=np.complex128), '0':np.array([1, 0], dtype=np.complex128)}
    psi_0 = [states_dict[p] for p in psi_0_list]
    # flatten psi_0 list, delete the sublists and keep the arrays
    psi_0 = [item for sublist in psi_0 for item in (sublist if isinstance(sublist, list) else [sublist])]
    psi_0_mps = MPS.from_product_state([SpinHalfSite(conserve=None)] * N, psi_0, "finite", dtype=np.complex128)
    return psi_0_mps


def choi_reshape(M):
    """
    Reshape the matrix M that is the tensor product of two 2**n x 2**n matrices
    into a matrix that corresponds to the dot product of the two vectorized matrices.
    """
    n = int(np.log2(M.shape[0]))
    M_reshaped = M.reshape(n,n,n,n)
    M_transposed = np.transpose(M_reshaped, axes=(0, 2, 1, 3))
    M_fin = M_transposed.reshape(n**2, n**2)
    return M_fin


def reconstruct_H(N, T, fiducial_states, probabilities):
    """
    Reconstruct the Hamiltonian from the given probabilities.
    """
    # build matrices of states and povms
    rho_vectors = [DensityMatrix(fiducial_state).dm.flatten() for fiducial_state in fiducial_states]
    povms_A = [rho_vector.conj() for rho_vector in rho_vectors]
    A = np.vstack(povms_A)
    B = np.column_stack(rho_vectors)

    # get process matrix
    G = np.linalg.pinv(A) @ probabilities @ np.linalg.pinv(B)

    # reshape it
    G_choi = choi_reshape(G)

    # build the U from probabilities as the rank-1 approximation of G_choi
    evals, evecs = np.linalg.eigh(G_choi)
    idx = np.argmax(np.abs(evals))
    lambda_ = evals[idx]
    v = evecs[:, idx]
    u_1 = np.sqrt(lambda_) * v

    # reshape u_1 to get the process matrix
    U_rec = u_1.reshape(2**(N), 2**(N))
    # project to the unitary group
    U_rec = complex2unitary(U_rec)
    if not np.allclose(np.eye(2**N), U_rec @ U_rec.conj().T):
        print("Warning: U_rec is not unitary")
    H_rec =  scipy.linalg.logm(U_rec) / (-1.j *T)

    return H_rec, G_choi


from scipy.linalg import inv, sqrtm
def complex2unitary( Z ):

    if Z.ndim == 3:
        for k, z in enumerate(Z):
            Z[k] = z @ inv( sqrtm( z.T.conj()@z ) )
    else:
        Z = Z @ inv( sqrtm( Z.T.conj()@Z ) )

    return Z 

def get_fiducial_states_arr(n, complete_set=False):
    """
    Generate fiducial states for tomography as array, useful to get 
    the coefficients.
    """
    sq_states = []
    # make single qubit states, then tensor them 
    psi_0 = np.array([1, 0], dtype=np.complex128)
    sq_states.append(psi_0)
    psi_1 = np.array([0, 1], dtype=np.complex128)
    sq_states.append(psi_1)
    psi_plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    sq_states.append(psi_plus)
    if complete_set:
        psi_minus = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)
        sq_states.append(psi_minus)
    psi_L = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    sq_states.append(psi_L)
    if complete_set:
        psi_R = np.array([1, -1j], dtype=np.complex128) / np.sqrt(2)
        sq_states.append(psi_R)

    # tensor them
    fiducial_states = []
    for state_combination in product(sq_states, repeat=n):
        psi_combined = state_combination[0]
        for psi in state_combination[1:]:
            psi_combined = np.kron(psi_combined, psi)
        fiducial_states.append(psi_combined)
    return fiducial_states

###############################################################################################################
# Hamiltonian model
###############################################################################################################

class CustomSpinModel(CouplingModel, NearestNeighborModel):
    """
    Custom spin model that combines CouplingModel and NearestNeighborModel from TeNPy.
    Allows flexible addition of onsite and coupling terms.
    """
    def __init__(self, lattice, h_strength=1.0, J_strength=1.0, seed=None, coeffs=None, ising_like=False):
        CouplingModel.__init__(self, lattice)
        self.h_strength = h_strength
        self.J_strength = J_strength
        self.seed = seed
        self._my_rng = np.random.default_rng(seed)
        self.N = lattice.N_sites
        self.str_ham = ''
        self.coeffs_vec = []
        self.coeffs = coeffs
        self.operators_vec = []
        if ising_like:
            self._build_ising_terms()
        else: 
            self._build_terms()
        self.H_bond = self.calc_H_bond()
        
    def _build_ising_terms(self):
        # 1/8 Z + 1/16 XX
        for i in range(self.N):
            self.add_onsite_term(1/8, i, 'Sigmaz')
            self.str_ham += f"1/8 * Sigmaz_{i} + "
            self.coeffs_vec.append(1/8)
            self.operators_vec.append('Sigmaz')

            if i == self.N - 1:
                continue
            self.add_coupling_term(1/16, i, i+1, 'Sigmax', 'Sigmax')
            self.str_ham += f"1/16 * Sigmax_{i} * Sigmax_{i+1} + "
            self.coeffs_vec.append(1/16)
            self.operators_vec.append('Sigmax_Sigmax')
        self.str_ham = self.str_ham[:-3]  # Remove the last ' + '

    def _build_terms(self):
        counter = 0
        for i in range(self.N):
            for op_str in ['Sigmax', 'Sigmay', 'Sigmaz']:
                if self.coeffs is not None:
                    coeff = self.coeffs[counter]
                    counter += 1
                else:
                    coeff = self._my_rng.uniform(-self.h_strength, self.h_strength)
                self.add_onsite_term(coeff, i, op_str)
                self.str_ham += f"{coeff} * {op_str}_{i} + "
                self.coeffs_vec.append(coeff)
                self.operators_vec.append(op_str)

            if i == self.N - 1:
                continue

            for op_str1 in ['Sigmax', 'Sigmay', 'Sigmaz']:
                for op_str2 in ['Sigmax', 'Sigmay', 'Sigmaz']:
                    if self.coeffs is not None:
                        coeff = self.coeffs[counter]
                        counter += 1
                    else:
                        coeff = self._my_rng.uniform(-self.J_strength, self.J_strength)
                    self.add_coupling_term(coeff, i, i+1, op_str1, op_str2)
                    self.str_ham += f"{coeff} * {op_str1}_{i} * {op_str2}_{i+1} + "
                    self.coeffs_vec.append(coeff)
                    self.operators_vec.append(f"{op_str1}_{op_str2}")

        self.str_ham = self.str_ham[:-3]  # Remove the last ' + '

    def calc_H_bond(self):
        # Use CouplingModel's method
        return CouplingModel.calc_H_bond(self)

    def calc_H_MPO(self):
        # Use CouplingModel's method
        return CouplingModel.calc_H_MPO(self)


###############################################################################################################
# Hamiltonian learning essentials
###############################################################################################################

class Hamiltonian:
    """
    Class representing a Hamiltonian.
    """
    def __init__(self, N, matrix=None, seed=None, ising_like=False):

        self.N = N
        self.h_space = HilbertSpace(N, 'spin')
        self.H = np.zeros((self.h_space.dimension, self.h_space.dimension), dtype=complex) if matrix is None else matrix
        self.str_ham = ''
        self.coeffs_vec = []
        self.operators_vec = []
        if seed is not None:
            np.random.seed(seed)
        
        if matrix is None:
            if ising_like:
                self.generate_ising_terms()
            else:
                self.generate_random_hamiltonian()
    def generate_ising_terms(self):
        """
        Generate an Ising-like Hamiltonian.
        """
        
        for i in range(self.N):
            operator = self.select_operator('Z', i)
            coeff = 1/8
            self.H += coeff * operator
            self.coeffs_vec.append(coeff)
            self.operators_vec.append(operator)
            self.str_ham += f"{coeff} * Z_{i} + "
            
            if i == self.N - 1:
                continue
            
            operator1 = self.select_operator('X', i)
            operator2 = self.select_operator('X', i + 1)
            coeff = 1/16
            self.H += coeff * (operator1 @ operator2)
            self.coeffs_vec.append(coeff)
            self.operators_vec.append(operator1 @ operator2)
            self.str_ham += f"{coeff} * X_{i} * X_{i+1} + "
        
        self.str_ham = self.str_ham[:-3]

    def generate_random_hamiltonian(self):
        """
        Generate a random Hamiltonian.
        """
        
        for i in range(self.N):
            for op_str in ['X', 'Y', 'Z']:
                operator = self.select_operator(op_str, i)
                coeff = np.random.uniform(-1, 1)
                self.H += coeff * operator
                self.coeffs_vec.append(coeff)
                self.operators_vec.append(operator)
                self.str_ham += f"{coeff} * {op_str}_{i} + "
                del operator
            
            if i == self.N - 1:
                continue
            
            for op_str1 in ['X', 'Y', 'Z']:
                for op_str2 in ['X', 'Y', 'Z']:
                    operator1 = self.select_operator(op_str1, i)
                    operator2 = self.select_operator(op_str2, i + 1)
                    coeff = np.random.uniform(-1, 1)
                    self.H += coeff * (operator1 @ operator2)
                    self.coeffs_vec.append(coeff)
                    self.operators_vec.append(operator1 @ operator2)
                    self.str_ham += f"{coeff} * {op_str1}_{i} * {op_str2}_{i+1} + "
                    del operator1, operator2
        
        self.str_ham = self.str_ham[:-3]
    
    def select_operator(self, string, idx):
        """
        Select the operator based on the given string.
        """
        if string == 'X':
            return self.h_space.X[idx]
        elif string == 'Y':
            return self.h_space.Y[idx]
        elif string == 'Z':
            return self.h_space.Z[idx]
        elif string == 'I':
            return np.eye(self.h_space.dimension)
        else:
            raise ValueError("Invalid string")
        

def hamiltonian_learning_w_QPT(N, T, fiducial_states, probabilities, return_G=False, ising_like=False):
    """
    Perform Hamiltonian learning with QPT.
    """
    H_rec, G_choi = reconstruct_H(N, T, fiducial_states, probabilities)
    H_trial = Hamiltonian(N, ising_like=ising_like)
    coeffs_rec = []
    for i in range(len(H_trial.coeffs_vec)):
        coeffs_rec.append(np.real(np.trace(H_trial.operators_vec[i] @ H_rec))/2**(N))
    if return_G:
        return coeffs_rec, G_choi
    else:
        return coeffs_rec
    

def combine_coeffs(coeffs_dict, ising_like=False):
    """
    Combine the coefficients from different patches.
    """
    def key_to_tuple(k):
        # split key in half
        mid = len(k) // 2
        k = (k[:mid], k[mid:])
        return int(k[0])

    sorted_keys = sorted(coeffs_dict.keys(), key=key_to_tuple)
    
    if len(sorted_keys) < 2:
        raise ValueError("Need at least two patches to combine coefficients.")
    
    combined = []
    first_key = sorted_keys[0]
    patch0 = coeffs_dict[first_key]
    if ising_like:
        # For Ising-like Hamiltonian
        combined.extend(patch0)
        for key in sorted_keys[1:]:
            patch = coeffs_dict[key][1:]
            combined.extend(patch)
        return combined
    combined.extend(patch0[:12])
    
    num_patches = len(sorted_keys)
    for i in range(1, num_patches):
        curr_key = sorted_keys[i]
        prev_key = sorted_keys[i-1]
        curr_patch = coeffs_dict[curr_key]
        prev_patch = coeffs_dict[prev_key]
        
        adjusted = list(np.array(curr_patch[:3]) - np.array(prev_patch[9:12]))
        combined.extend(adjusted)
        
        if i < num_patches - 1:
            combined.extend(curr_patch[3:12])
        else:
            combined.extend(curr_patch[3:])
            
    return combined

###############################################################################################################
# main
###############################################################################################################

def main():
    parser = argparse.ArgumentParser(description="Run Zeno script with specified parameters.")
    parser.add_argument('--N', type=int, default=7, help='Number of spins')
    parser.add_argument('--N_QPT', type=int, default=2, help='Number of qubits for QPT')
    parser.add_argument('--T', type=float, default=1e-2, help='Total evolution time')
    parser.add_argument('--n_steps', type=float, default=10, help='Number of steps for evolution')
    parser.add_argument('--N_shots', type=int, default=100000, help='Number of shots for simulation')
    parser.add_argument('--plot', type=bool, default=True, help='Flag to plot the results')
    parser.add_argument('--save', type=bool, default=False, help='Flag to save the results to a text file')
    parser.add_argument('--seed', type=int, default=43, help='Random seed')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--ising_like', type=bool, default=False, help='Use Ising-like Hamiltonian')
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(args, key, value)

    N = args.N
    N_QPT = args.N_QPT
    T = args.T
    n_steps = args.n_steps
    dt = T/n_steps
    N_shots = args.N_shots
    plot = args.plot
    save = args.save
    seed = args.seed
    ising_like = args.ising_like

    np.random.seed(seed)

    # tomography essentials (on 2 spins)
    fiducial_states = get_fiducial_states(N_QPT)
    probabilities = {}
    for i in range(N-1):
            key = f"{i}{i+1}"
            probabilities[key] = np.zeros((len(get_qpt_keys(2, exte=True)), len(fiducial_states)), dtype=complex)

    site = SpinHalfSite(conserve=None)
    # Create a chain lattice with N sites
    lattice = Chain(N, site)
    coeffs_vec = np.random.uniform(-1, 1, size=(12*N-9))
    spin_model = CustomSpinModel(lattice, h_strength=1.0, J_strength=1.0, seed=seed, coeffs=coeffs_vec, ising_like=ising_like)

    tebd_params_ev = {
        'N_steps': 1,
        'dt': dt,
        'order': 4,
        'trunc_params': {'chi_max': 16, 'svd_min': 1.e-12}
    }   

    reshaping_steps = [0, 1, 2] if N > 3 else [0, 1]
    for reshaping_step in reshaping_steps:
        # generate psi_0_list
        psi_0_list = generate_psi0_list(N, reshaping_step)
        U_kick = generate_U_kick_from_list(psi_0_list) 
        for j, fiducial_state in tqdm(enumerate(fiducial_states)):
            # generate psi_0
            psi_0 = generate_psi0_fromlist(psi_0_list, fiducial_state, N)
            psi_zeno = psi_0
            for _ in range(n_steps):
                eng = tebd.TEBDEngine(psi_zeno, spin_model, tebd_params_ev)
                eng.run()
                psi_zeno = LocalProduct(psi_zeno, U_kick)  
            # apply U_kick dagger n_steps times
            U_kick_dagger = U_kick # for Z kick it is true
            for _ in range(n_steps):
                psi_zeno = LocalProduct(psi_zeno, U_kick_dagger)
            # get probabilities
            probs_tot = get_probabilities(psi_0_list, psi_zeno, get_qpt_keys(2, exte=True), N_shots=N_shots)
            required_keys = get_qpt_keys(2, exte=True)
            for qubit_key in probs_tot.keys():
                for i, meas_key in enumerate(required_keys):
                    p_ij = probs_tot[qubit_key][meas_key]
                    probabilities[qubit_key][i, j] = p_ij
    coeffs_rec = {}
    choi_matrices = {}
    fiducial_states_arr = get_fiducial_states_arr(N_QPT)
    for key in probabilities.keys():
        coeffs, choi = hamiltonian_learning_w_QPT(N_QPT, T, fiducial_states_arr, probabilities[key], return_G=True, ising_like=ising_like)
        coeffs_rec[key] = coeffs
        choi_matrices[key] = choi
    coeffs_rec_tot = combine_coeffs(coeffs_rec, ising_like)
    coeffs_vec = spin_model.coeffs_vec
    if save:
        with open('results.txt', 'w') as f:
            f.write("Reconstructed Coefficients:\n")
            f.write(repr(coeffs_rec_tot) + "\n")
            f.write("Real Coefficients:\n")
            f.write(repr(coeffs_vec) + "\n")
            f.write("Differences:\n")
            f.write(repr(np.abs(np.array(coeffs_rec_tot) - np.array(coeffs_vec))) + "\n")
            f.write("Choi Matrices:\n")
            for key, choi in choi_matrices.items():
                f.write(f"{key}:\n")
                f.write(str(choi) + "\n")

    if plot:
        # make a nice plot where we show the quality of the reconstruction, plot the difference between the coefficients
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(len(coeffs_rec_tot)), np.abs(np.array(coeffs_rec_tot) - np.array(coeffs_vec)), 'o', label='Difference')
        plt.ylim(1e-10, 10)
        plt.xlabel('Term')
        plt.ylabel('Difference')
        plt.yscale('log')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
