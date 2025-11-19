"""
Convolutional Coding Simulation - Alien Civilization 3

Implements convolutional encoding and Viterbi decoding for forward error correction.
Demonstrates how convolutional codes protect data against channel errors using
trellis-based encoding/decoding.

Educational Goals:
- Understand convolutional code structure
- Learn Viterbi algorithm for maximum likelihood decoding
- Visualize trellis diagrams and path metrics
- Explore code rate and constraint length trade-offs
- Analyze coding gain vs uncoded transmission

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


class ConvolutionalCodingSimulation(SimulationModule):
    """Convolutional encoder and Viterbi decoder simulation.

    Implements rate 1/2 convolutional code with configurable constraint length.
    Uses Viterbi algorithm for soft-decision decoding.

    Example:
        >>> sim = ConvolutionalCodingSimulation()
        >>> params = {'constraint_length': 7, 'num_bits': 100, 'snr_db': 3.0}
        >>> result = sim.run_simulation(params)
    """

    def get_name(self) -> str:
        return "Convolutional Coding (Civilization 3)"

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'constraint_length': 7,    # K=7 (industry standard)
            'num_bits': 100,           # Information bits
            'snr_db': 3.0,             # SNR for coded transmission
            'code_rate': 0.5,          # Rate 1/2 (2 output bits per input bit)
            'generator_polynomials': [0o171, 0o133]  # G1, G2 in octal
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            'constraint_length': {
                'type': 'int',
                'description': 'Constraint length K (memory length + 1)',
                'min': 3,
                'max': 9,
                'default': 7,
                'required': True,
                'choices': [3, 5, 7, 9]
            },
            'num_bits': {
                'type': 'int',
                'description': 'Number of information bits',
                'min': 10,
                'max': 1000,
                'default': 100,
                'required': True
            },
            'snr_db': {
                'type': 'float',
                'description': 'SNR in dB for coded bits',
                'min': -5.0,
                'max': 15.0,
                'default': 3.0,
                'required': True
            }
        }

    def _convolutional_encode(
        self,
        bits: np.ndarray,
        constraint_length: int,
        generators: List[int]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Encode bits using convolutional code.

        Args:
            bits: Input bit stream
            constraint_length: K (memory + 1)
            generators: Generator polynomials in integer form

        Returns:
            Tuple of (coded_bits, state_history)
        """
        # Initialize shift register
        memory = constraint_length - 1
        shift_reg = np.zeros(constraint_length, dtype=int)

        coded_bits = []
        state_history = []

        # Encode each bit
        for bit in bits:
            # Shift in new bit
            shift_reg = np.roll(shift_reg, 1)
            shift_reg[0] = bit

            # Generate output bits
            for gen in generators:
                # Convert generator to binary and apply
                gen_bits = [(gen >> i) & 1 for i in range(constraint_length)]
                output = np.sum(shift_reg * gen_bits) % 2
                coded_bits.append(output)

            # Record state
            state_history.append(shift_reg.copy())

        # Flush encoder (feed zeros to clear state)
        for _ in range(memory):
            shift_reg = np.roll(shift_reg, 1)
            shift_reg[0] = 0

            for gen in generators:
                gen_bits = [(gen >> i) & 1 for i in range(constraint_length)]
                output = np.sum(shift_reg * gen_bits) % 2
                coded_bits.append(output)

            state_history.append(shift_reg.copy())

        return np.array(coded_bits), state_history

    def _viterbi_decode(
        self,
        received_symbols: np.ndarray,
        constraint_length: int,
        generators: List[int]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Decode using Viterbi algorithm.

        Args:
            received_symbols: Soft-decision received values
            constraint_length: K
            generators: Generator polynomials

        Returns:
            Tuple of (decoded_bits, trellis_metrics)
        """
        memory = constraint_length - 1
        num_states = 2 ** memory
        n_outputs = len(generators)

        # Initialize path metrics (start in state 0)
        path_metrics = np.full(num_states, np.inf)
        path_metrics[0] = 0.0

        # Traceback storage
        paths = [[] for _ in range(num_states)]
        trellis_data = []

        # Process received symbols (in pairs for rate 1/2)
        num_stages = len(received_symbols) // n_outputs

        for stage in range(num_stages):
            new_metrics = np.full(num_states, np.inf)
            new_paths = [[] for _ in range(num_states)]
            stage_info = {'stage': stage, 'transitions': []}

            # For each current state
            for curr_state in range(num_states):
                if path_metrics[curr_state] == np.inf:
                    continue

                # Try both input bits (0 and 1)
                for input_bit in [0, 1]:
                    # Compute next state
                    next_state = ((curr_state << 1) | input_bit) & (num_states - 1)

                    # Compute expected output
                    shift_reg = np.zeros(constraint_length, dtype=int)
                    shift_reg[0] = input_bit
                    for i in range(memory):
                        shift_reg[i+1] = (curr_state >> (memory-1-i)) & 1

                    expected = []
                    for gen in generators:
                        gen_bits = [(gen >> i) & 1 for i in range(constraint_length)]
                        output = np.sum(shift_reg * gen_bits) % 2
                        expected.append(1.0 if output else -1.0)

                    # Calculate branch metric (Euclidean distance)
                    received = received_symbols[stage*n_outputs:(stage+1)*n_outputs]
                    branch_metric = np.sum((received - expected)**2)

                    # Update path metric
                    candidate = path_metrics[curr_state] + branch_metric

                    if candidate < new_metrics[next_state]:
                        new_metrics[next_state] = candidate
                        new_paths[next_state] = paths[curr_state] + [input_bit]

                    stage_info['transitions'].append({
                        'from': curr_state,
                        'to': next_state,
                        'input': input_bit,
                        'metric': candidate
                    })

            path_metrics = new_metrics
            paths = new_paths
            trellis_data.append(stage_info)

        # Find best final state (should be 0 after flush)
        best_state = np.argmin(path_metrics)
        decoded_bits = np.array(paths[best_state])

        return decoded_bits, trellis_data

    def run_simulation(self, parameters: Dict[str, Any]) -> SimulationResult:
        """Execute convolutional coding simulation.

        Args:
            parameters: Simulation parameters

        Returns:
            SimulationResult with coding performance data
        """
        start_time = datetime.now()
        start_perf = time.perf_counter()

        try:
            # Validate
            self.validate_parameters(parameters)

            # Extract parameters
            K = parameters['constraint_length']
            num_bits = parameters['num_bits']
            snr_db = parameters['snr_db']
            generators = parameters.get('generator_polynomials', [0o171, 0o133])

            # Generate random information bits
            info_bits = np.random.randint(0, 2, num_bits)

            # Encode
            coded_bits, state_history = self._convolutional_encode(
                info_bits, K, generators
            )

            # Modulate (BPSK: 0->-1, 1->+1)
            tx_symbols = 2.0 * coded_bits - 1.0

            # Add AWGN
            snr_linear = 10 ** (snr_db / 10.0)
            noise_power = 1.0 / snr_linear
            noise = np.sqrt(noise_power / 2) * np.random.randn(len(tx_symbols))
            rx_symbols = tx_symbols + noise

            # Decode
            decoded_bits, trellis_data = self._viterbi_decode(
                rx_symbols, K, generators
            )

            # Calculate BER
            # Remove tail bits used for flushing
            decoded_info = decoded_bits[:num_bits]
            bit_errors = np.sum(info_bits != decoded_info)
            ber = bit_errors / num_bits

            # Compare with uncoded performance (for reference)
            uncoded_symbols = 2.0 * info_bits - 1.0
            uncoded_rx = uncoded_symbols + np.sqrt(noise_power/2) * np.random.randn(num_bits)
            uncoded_decisions = (uncoded_rx > 0).astype(int)
            uncoded_errors = np.sum(info_bits != uncoded_decisions)
            uncoded_ber = uncoded_errors / num_bits

            # Calculate coding gain
            if uncoded_ber > 0 and ber > 0:
                coding_gain_db = 10 * np.log10(uncoded_ber / ber)
            else:
                coding_gain_db = float('inf') if ber == 0 else 0.0

            # Execution time
            execution_time = (time.perf_counter() - start_perf) * 1000

            result_data = {
                'info_bits': info_bits,
                'coded_bits': coded_bits,
                'tx_symbols': tx_symbols,
                'rx_symbols': rx_symbols,
                'decoded_bits': decoded_info,
                'state_history': state_history,
                'trellis_data': trellis_data,
                'ber': ber,
                'uncoded_ber': uncoded_ber,
                'bit_errors': int(bit_errors),
                'uncoded_errors': int(uncoded_errors),
                'coding_gain_db': coding_gain_db,
                'noise_power': noise_power
            }

            metadata = {
                'module': self.get_name(),
                'constraint_length': K,
                'code_rate': 0.5,
                'generator_polynomials': generators,
                'num_states': 2 ** (K - 1)
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
    print("=== Convolutional Coding Simulation Demo ===\n")

    sim = ConvolutionalCodingSimulation()
    print(f"Simulation: {sim.get_name()}\n")

    # Test different SNR levels
    for snr_db in [0.0, 3.0, 6.0]:
        params = {
            'constraint_length': 7,
            'num_bits': 100,
            'snr_db': snr_db
        }

        result = sim.run_simulation(params)

        if result.success:
            print(f"SNR: {snr_db} dB")
            print(f"  Coded BER: {result.data['ber']:.6f}")
            print(f"  Uncoded BER: {result.data['uncoded_ber']:.6f}")
            print(f"  Coding Gain: {result.data['coding_gain_db']:.2f} dB")
            print(f"  Bit Errors: {result.data['bit_errors']}/{len(result.data['info_bits'])}")
            print()
        else:
            print(f"  Failed: {result.error_message}\n")

    print("Convolutional coding simulation ready!")
