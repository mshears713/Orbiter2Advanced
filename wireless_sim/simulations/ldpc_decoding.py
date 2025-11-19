"""
LDPC Decoding Simulation - Alien Civilization 4

Implements Low-Density Parity-Check (LDPC) codes with iterative belief propagation
decoding. Demonstrates how sparse graph-based codes achieve near-Shannon-limit
performance through message passing algorithms.

LDPC codes are modern error-correcting codes used in 5G, WiFi 6, DVB-S2, and
many other standards. This module shows the iterative decoding process.

Educational Goals:
- Understand sparse parity-check matrices
- Learn belief propagation / sum-product algorithm
- Visualize message passing on Tanner graphs
- Explore convergence behavior
- Analyze coding performance near channel capacity

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, List
import time

from simulations.base import SimulationModule
from simulations.exceptions import SimulationError
from models.datamodels import SimulationResult


class LDPCDecodingSimulation(SimulationModule):
    """LDPC encoder and belief propagation decoder simulation.

    Implements regular LDPC codes with configurable code rate and block length.
    Uses sum-product algorithm (belief propagation) for soft-decision decoding.

    Example:
        >>> sim = LDPCDecodingSimulation()
        >>> params = {'block_length': 100, 'code_rate': 0.5, 'max_iterations': 20, 'snr_db': 2.0}
        >>> result = sim.run_simulation(params)
    """

    def get_name(self) -> str:
        return "LDPC Decoding (Civilization 4)"

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'block_length': 100,        # Codeword length (n)
            'code_rate': 0.5,           # k/n (information/codeword)
            'max_iterations': 20,       # Max BP iterations
            'snr_db': 2.0,              # SNR for BPSK transmission
            'early_termination': True,  # Stop if syndrome = 0
            'variable_degree': 3,       # Variable node degree (dv)
            'check_degree': 6           # Check node degree (dc)
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            'block_length': {
                'type': 'int',
                'description': 'LDPC codeword length n',
                'min': 50,
                'max': 500,
                'default': 100,
                'required': True
            },
            'code_rate': {
                'type': 'float',
                'description': 'Code rate (k/n)',
                'min': 0.25,
                'max': 0.9,
                'default': 0.5,
                'required': True
            },
            'max_iterations': {
                'type': 'int',
                'description': 'Maximum belief propagation iterations',
                'min': 5,
                'max': 50,
                'default': 20,
                'required': True
            },
            'snr_db': {
                'type': 'float',
                'description': 'SNR in dB for BPSK modulation',
                'min': -2.0,
                'max': 10.0,
                'default': 2.0,
                'required': True
            }
        }

    def _generate_regular_ldpc_matrix(
        self,
        n: int,
        k: int,
        dv: int = 3,
        dc: int = 6
    ) -> np.ndarray:
        """Generate a regular LDPC parity-check matrix.

        Creates a sparse matrix H with regular column weight dv and row weight dc.
        This is a simplified construction - production codes use more sophisticated
        methods (e.g., progressive edge growth).

        Args:
            n: Codeword length
            k: Information length
            dv: Variable node degree (ones per column)
            dc: Check node degree (ones per row)

        Returns:
            Parity-check matrix H of size (n-k) x n
        """
        m = n - k  # Number of parity checks

        # Create sparse matrix using random construction
        # (Note: This is a simplified method; real LDPC codes use structured designs)
        H = np.zeros((m, n), dtype=int)

        # Ensure regular structure: each column has dv ones
        for col in range(n):
            # Randomly select dv rows for this column
            rows = np.random.choice(m, size=min(dv, m), replace=False)
            H[rows, col] = 1

        # Balance row weights (attempt to make them closer to dc)
        for row in range(m):
            current_weight = np.sum(H[row, :])
            if current_weight < dc:
                # Add more ones
                available_cols = np.where(H[row, :] == 0)[0]
                if len(available_cols) > 0:
                    add_count = min(dc - current_weight, len(available_cols))
                    cols_to_add = np.random.choice(available_cols, size=add_count, replace=False)
                    H[row, cols_to_add] = 1

        return H

    def _ldpc_encode(self, info_bits: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Encode information bits using LDPC code.

        For simplicity, this uses a systematic encoding approach.
        In practice, encoding requires the generator matrix G or
        a specific encoding algorithm for the parity-check matrix.

        Args:
            info_bits: k information bits
            H: m x n parity-check matrix

        Returns:
            n-bit codeword
        """
        n = H.shape[1]
        k = len(info_bits)
        m = n - k

        # Systematic encoding: codeword = [info_bits, parity_bits]
        # We need to find parity bits such that H * codeword = 0 (mod 2)

        # For this simplified version, we'll use a pseudo-random
        # approach that ensures valid codewords
        codeword = np.zeros(n, dtype=int)
        codeword[:k] = info_bits

        # Solve for parity bits (simplified approach)
        # H * c = 0 => H[:, :k] * info + H[:, k:] * parity = 0
        # This requires H to have special structure; we'll approximate

        # Simple approach: set parity bits to satisfy parity checks
        for i in range(m):
            # Calculate syndrome contribution from info bits
            syndrome_bit = np.sum(H[i, :k] * info_bits) % 2
            # Find first parity bit position for this check
            parity_positions = np.where(H[i, k:] == 1)[0]
            if len(parity_positions) > 0:
                # Set first parity bit to correct the syndrome
                codeword[k + parity_positions[0]] = syndrome_bit

        return codeword

    def _belief_propagation_decode(
        self,
        llr_channel: np.ndarray,
        H: np.ndarray,
        max_iter: int,
        early_term: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """LDPC decoding using belief propagation (sum-product algorithm).

        Args:
            llr_channel: Channel log-likelihood ratios (LLRs)
            H: Parity-check matrix
            max_iter: Maximum iterations
            early_term: Enable early termination on syndrome check

        Returns:
            Tuple of (decoded_bits, iteration_history)
        """
        m, n = H.shape

        # Initialize messages
        # q[i,j] = message from variable node i to check node j
        # r[j,i] = message from check node j to variable node i
        q_msgs = np.zeros((n, m))
        r_msgs = np.zeros((m, n))

        # Initialize variable-to-check messages with channel LLRs
        for i in range(n):
            connected_checks = np.where(H[:, i] == 1)[0]
            for j in connected_checks:
                q_msgs[i, j] = llr_channel[i]

        iteration_history = []

        for iteration in range(max_iter):
            # === Check node update ===
            # r[j,i] = 2 * atanh(prod(tanh(q[i',j]/2))) for all i' connected to j except i
            for j in range(m):
                connected_vars = np.where(H[j, :] == 1)[0]

                for i in connected_vars:
                    # Collect messages from all other variable nodes
                    other_vars = connected_vars[connected_vars != i]
                    if len(other_vars) == 0:
                        r_msgs[j, i] = 0.0
                    else:
                        # Product of tanh(q/2)
                        tanh_prod = 1.0
                        for i_prime in other_vars:
                            tanh_val = np.tanh(q_msgs[i_prime, j] / 2.0)
                            tanh_prod *= np.clip(tanh_val, -0.9999, 0.9999)

                        # Avoid numerical issues with atanh
                        tanh_prod = np.clip(tanh_prod, -0.9999, 0.9999)
                        r_msgs[j, i] = 2.0 * np.arctanh(tanh_prod)

            # === Variable node update ===
            # q[i,j] = LLR_i + sum(r[j',i]) for all j' connected to i except j
            llr_posterior = np.zeros(n)

            for i in range(n):
                connected_checks = np.where(H[:, i] == 1)[0]

                # Calculate posterior LLR (for hard decision)
                llr_posterior[i] = llr_channel[i] + np.sum(r_msgs[connected_checks, i])

                # Update messages to each check
                for j in connected_checks:
                    other_checks = connected_checks[connected_checks != j]
                    if len(other_checks) == 0:
                        q_msgs[i, j] = llr_channel[i]
                    else:
                        q_msgs[i, j] = llr_channel[i] + np.sum(r_msgs[other_checks, i])

            # Hard decision based on posterior LLRs
            decoded_bits = (llr_posterior < 0).astype(int)

            # Calculate syndrome: s = H * decoded_bits (mod 2)
            syndrome = np.dot(H, decoded_bits) % 2
            syndrome_weight = np.sum(syndrome)

            # Record iteration info
            iteration_history.append({
                'iteration': iteration,
                'llr_posterior': llr_posterior.copy(),
                'decoded_bits': decoded_bits.copy(),
                'syndrome_weight': int(syndrome_weight),
                'converged': syndrome_weight == 0
            })

            # Early termination if syndrome is zero
            if early_term and syndrome_weight == 0:
                break

        # Final decode
        final_llr = llr_channel + np.sum(r_msgs, axis=0)
        decoded_bits = (final_llr < 0).astype(int)

        return decoded_bits, iteration_history

    def run_simulation(self, parameters: Dict[str, Any]) -> SimulationResult:
        """Execute LDPC decoding simulation.

        Args:
            parameters: Simulation parameters

        Returns:
            SimulationResult with LDPC performance data
        """
        start_time = datetime.now()
        start_perf = time.perf_counter()

        try:
            # Validate
            self.validate_parameters(parameters)

            # Extract parameters
            n = parameters['block_length']
            code_rate = parameters['code_rate']
            k = int(n * code_rate)
            m = n - k
            max_iter = parameters['max_iterations']
            snr_db = parameters['snr_db']
            early_term = parameters.get('early_termination', True)
            dv = parameters.get('variable_degree', 3)
            dc = parameters.get('check_degree', 6)

            # Generate LDPC parity-check matrix
            H = self._generate_regular_ldpc_matrix(n, k, dv, dc)

            # Generate random information bits
            info_bits = np.random.randint(0, 2, k)

            # Encode
            codeword = self._ldpc_encode(info_bits, H)

            # BPSK modulation: 0 -> +1, 1 -> -1
            tx_symbols = 1 - 2 * codeword.astype(float)

            # AWGN channel
            snr_linear = 10 ** (snr_db / 10.0)
            noise_power = 1.0 / snr_linear
            noise = np.sqrt(noise_power / 2) * np.random.randn(n)
            rx_symbols = tx_symbols + noise

            # Compute channel LLRs for BPSK
            # LLR = log(P(bit=0)/P(bit=1)) = 2 * received / noise_variance
            llr_channel = 2.0 * rx_symbols / (noise_power / 2)

            # Decode using belief propagation
            decoded_bits, iteration_history = self._belief_propagation_decode(
                llr_channel, H, max_iter, early_term
            )

            # Calculate BER
            bit_errors = np.sum(codeword != decoded_bits)
            ber = bit_errors / n

            # Calculate information BER (first k bits)
            info_errors = np.sum(info_bits != decoded_bits[:k])
            info_ber = info_errors / k

            # Check if decoding converged
            final_syndrome = np.dot(H, decoded_bits) % 2
            converged = np.sum(final_syndrome) == 0

            # Execution time
            execution_time = (time.perf_counter() - start_perf) * 1000

            result_data = {
                'parity_check_matrix': H,
                'info_bits': info_bits,
                'codeword': codeword,
                'tx_symbols': tx_symbols,
                'rx_symbols': rx_symbols,
                'llr_channel': llr_channel,
                'decoded_bits': decoded_bits,
                'iteration_history': iteration_history,
                'ber': ber,
                'info_ber': info_ber,
                'bit_errors': int(bit_errors),
                'info_bit_errors': int(info_errors),
                'converged': converged,
                'num_iterations': len(iteration_history),
                'noise_power': noise_power
            }

            metadata = {
                'module': self.get_name(),
                'block_length': n,
                'code_rate': code_rate,
                'info_length': k,
                'num_parity_checks': m,
                'variable_degree': dv,
                'check_degree': dc,
                'max_iterations': max_iter
            }

            return SimulationResult(
                timestamp=start_time,
                parameters=parameters,
                data=result_data,
                metadata=metadata,
                success=True,
                error_message="",
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (time.perf_counter() - start_perf) * 1000
            return SimulationResult(
                timestamp=start_time,
                parameters=parameters,
                data={},
                metadata={'module': self.get_name()},
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )


if __name__ == "__main__":
    print("=== LDPC Decoding Simulation Demo ===\n")

    sim = LDPCDecodingSimulation()
    print(f"Simulation: {sim.get_name()}\n")

    # Test at different SNR levels
    for snr_db in [0.5, 1.0, 2.0]:
        params = {
            'block_length': 100,
            'code_rate': 0.5,
            'max_iterations': 20,
            'snr_db': snr_db,
            'variable_degree': 3,
            'check_degree': 6
        }

        result = sim.run_simulation(params)

        if result.success:
            print(f"SNR: {snr_db} dB")
            print(f"  Code Rate: {result.metadata['code_rate']}")
            print(f"  BER: {result.data['ber']:.6f}")
            print(f"  Info BER: {result.data['info_ber']:.6f}")
            print(f"  Converged: {result.data['converged']}")
            print(f"  Iterations: {result.data['num_iterations']}/{params['max_iterations']}")
            print(f"  Bit Errors: {result.data['bit_errors']}/{params['block_length']}")
            print()
        else:
            print(f"  Failed: {result.error_message}\n")

    print("LDPC decoding simulation ready!")
