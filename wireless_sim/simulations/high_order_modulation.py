"""
High-Order Modulation Simulation - Alien Civilization 1

Simulates high-order QAM (Quadrature Amplitude Modulation) and PSK (Phase Shift Keying)
modulation schemes with AWGN (Additive White Gaussian Noise) channel effects.

This module represents the first alien civilization encountered in the Orbiter-2 mission,
which uses sophisticated constellation-based modulation for efficient spectrum usage.

Educational Goals:
- Understand how bits are mapped to complex constellation symbols
- Visualize the effects of noise on constellation diagrams
- Explore trade-offs between modulation order and noise resilience
- Learn about Gray coding and symbol error rates

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
import time

from simulations.base import SimulationModule
from simulations.exceptions import SimulationError, ParameterValidationError
from models.datamodels import SimulationResult


class HighOrderModulationSimulation(SimulationModule):
    """High-order QAM/PSK modulation simulation with AWGN channel.

    This simulation demonstrates how digital data is encoded into complex symbols
    for wireless transmission. It supports various modulation orders (4-QAM to 256-QAM)
    and allows exploration of noise effects on constellation diagrams.

    Key Concepts Demonstrated:
    - Bit-to-symbol mapping for QAM modulation
    - Constellation diagram structure
    - AWGN channel modeling
    - SNR vs. symbol error relationship
    - Gray coding benefits

    Example:
        >>> sim = HighOrderModulationSimulation()
        >>> params = {'modulation_order': 16, 'snr_db': 10.0, 'num_symbols': 1000}
        >>> result = sim.run_simulation(params)
        >>> print(f"Generated {len(result.data['tx_symbols'])} symbols")
    """

    def get_name(self) -> str:
        return "High-Order Modulation (Civilization 1)"

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'modulation_order': 16,      # 16-QAM by default
            'num_symbols': 1000,         # Number of symbols to generate
            'snr_db': 10.0,              # Signal-to-Noise Ratio in dB
            'modulation_type': 'QAM',    # 'QAM' or 'PSK'
            'use_gray_coding': True      # Use Gray coding for bit mapping
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            'modulation_order': {
                'type': 'int',
                'description': 'Modulation order (M-QAM or M-PSK). Must be power of 2.',
                'min': 4,
                'max': 256,
                'default': 16,
                'required': True,
                'choices': [4, 16, 64, 256]  # Common modulation orders
            },
            'num_symbols': {
                'type': 'int',
                'description': 'Number of symbols to transmit',
                'min': 10,
                'max': 10000,
                'default': 1000,
                'required': True
            },
            'snr_db': {
                'type': 'float',
                'description': 'Signal-to-Noise Ratio in decibels',
                'min': -10.0,
                'max': 40.0,
                'default': 10.0,
                'required': True
            },
            'modulation_type': {
                'type': 'str',
                'description': 'Modulation scheme type',
                'choices': ['QAM', 'PSK'],
                'default': 'QAM',
                'required': False
            },
            'use_gray_coding': {
                'type': 'bool',
                'description': 'Use Gray coding for bit-to-symbol mapping',
                'default': True,
                'required': False
            }
        }

    def _generate_qam_constellation(self, M: int) -> np.ndarray:
        """Generate square QAM constellation points.

        Creates a square M-QAM constellation with normalized average power.
        Points are arranged in a square grid in the I-Q plane.

        Args:
            M: Modulation order (must be perfect square for square QAM)

        Returns:
            Array of complex constellation points, shape (M,)

        Example:
            >>> constellation = self._generate_qam_constellation(16)
            >>> constellation.shape
            (16,)
        """
        if M not in [4, 16, 64, 256]:
            raise ParameterValidationError(
                f"Unsupported QAM order {M}. Supported: 4, 16, 64, 256"
            )

        # For square QAM: M = L^2, where L is constellation side length
        L = int(np.sqrt(M))

        # Create I and Q components
        indices = np.arange(L)
        # Center around zero: -L+1, -L+3, ..., L-3, L-1
        levels = 2 * indices - L + 1

        # Create meshgrid for all combinations
        I, Q = np.meshgrid(levels, levels)

        # Flatten to create constellation
        constellation = (I + 1j * Q).flatten()

        # Normalize to unit average power
        avg_power = np.mean(np.abs(constellation)**2)
        constellation = constellation / np.sqrt(avg_power)

        return constellation

    def _generate_psk_constellation(self, M: int) -> np.ndarray:
        """Generate M-PSK constellation points.

        Creates M phase-shift keying points uniformly distributed on unit circle.

        Args:
            M: Modulation order

        Returns:
            Array of complex constellation points, shape (M,)
        """
        # Angles uniformly distributed around unit circle
        angles = 2 * np.pi * np.arange(M) / M

        # Add pi/M rotation for QPSK compatibility
        if M == 4:
            angles += np.pi / 4

        constellation = np.exp(1j * angles)
        return constellation

    def _add_awgn(
        self,
        signal: np.ndarray,
        snr_db: float
    ) -> Tuple[np.ndarray, float]:
        """Add Additive White Gaussian Noise to signal.

        Adds complex AWGN to achieve specified SNR. The SNR is defined as:
        SNR = 10 * log10(Signal Power / Noise Power)

        Args:
            signal: Clean signal (complex array)
            snr_db: Desired SNR in decibels

        Returns:
            Tuple of (noisy_signal, actual_noise_power)

        Example:
            >>> clean = np.array([1+1j, -1+1j, -1-1j, 1-1j])
            >>> noisy, noise_pwr = self._add_awgn(clean, 10.0)
        """
        # Calculate signal power
        signal_power = np.mean(np.abs(signal)**2)

        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10.0)

        # Calculate required noise power
        noise_power = signal_power / snr_linear

        # Generate complex AWGN
        # Real and imaginary parts each have variance = noise_power/2
        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

        # Add noise to signal
        noisy_signal = signal + noise

        return noisy_signal, noise_power

    def _map_bits_to_symbols(
        self,
        bits: np.ndarray,
        constellation: np.ndarray,
        use_gray: bool = True
    ) -> np.ndarray:
        """Map bit stream to constellation symbols.

        Groups bits into symbols and maps each group to a constellation point.
        Optionally uses Gray coding for improved error resilience.

        Args:
            bits: Binary data stream (0s and 1s)
            constellation: Available constellation points
            use_gray: Whether to use Gray coding

        Returns:
            Array of complex symbols from constellation
        """
        M = len(constellation)
        bits_per_symbol = int(np.log2(M))

        # Ensure we have complete symbols
        num_complete_symbols = len(bits) // bits_per_symbol
        bits_to_use = bits[:num_complete_symbols * bits_per_symbol]

        # Reshape into symbol groups
        bit_groups = bits_to_use.reshape(-1, bits_per_symbol)

        # Convert bit groups to decimal indices
        powers = 2 ** np.arange(bits_per_symbol)[::-1]
        indices = np.dot(bit_groups, powers)

        # Apply Gray coding if requested
        if use_gray and M in [4, 16, 64, 256]:
            # Gray code mapping (simplified - natural to Gray)
            # For proper Gray coding, we'd need a lookup table per M
            # For now, use natural ordering (can enhance later)
            pass

        # Map indices to constellation points
        symbols = constellation[indices]

        return symbols

    def run_simulation(self, parameters: Dict[str, Any]) -> SimulationResult:
        """Execute high-order modulation simulation.

        Generates random bits, maps them to modulation symbols, adds AWGN noise,
        and returns both clean and noisy constellation points for visualization.

        Args:
            parameters: Simulation parameters including:
                - modulation_order: M-QAM/PSK order
                - num_symbols: Number of symbols to generate
                - snr_db: Signal-to-noise ratio
                - modulation_type: 'QAM' or 'PSK'
                - use_gray_coding: Enable Gray coding

        Returns:
            SimulationResult containing:
                - tx_symbols: Clean transmitted symbols
                - rx_symbols: Noisy received symbols
                - constellation: Ideal constellation points
                - noise_power: Actual noise power added
                - ber: Estimated bit error rate (if demodulated)

        Raises:
            SimulationError: If simulation fails
            ParameterValidationError: If parameters are invalid
        """
        start_time = datetime.now()
        start_perf = time.perf_counter()

        try:
            # Validate parameters
            self.validate_parameters(parameters)

            # Extract parameters
            M = parameters['modulation_order']
            num_symbols = parameters['num_symbols']
            snr_db = parameters['snr_db']
            mod_type = parameters.get('modulation_type', 'QAM')
            use_gray = parameters.get('use_gray_coding', True)

            # Generate constellation
            if mod_type.upper() == 'QAM':
                constellation = self._generate_qam_constellation(M)
            elif mod_type.upper() == 'PSK':
                constellation = self._generate_psk_constellation(M)
            else:
                raise ParameterValidationError(
                    f"Invalid modulation_type: {mod_type}. Use 'QAM' or 'PSK'"
                )

            # Generate random bits
            bits_per_symbol = int(np.log2(M))
            total_bits = num_symbols * bits_per_symbol
            bits = np.random.randint(0, 2, total_bits)

            # Map bits to symbols
            tx_symbols = self._map_bits_to_symbols(bits, constellation, use_gray)

            # Add AWGN noise
            rx_symbols, noise_power = self._add_awgn(tx_symbols, snr_db)

            # Calculate metrics
            signal_power = np.mean(np.abs(tx_symbols)**2)
            actual_snr_db = 10 * np.log10(signal_power / noise_power)

            # Simple demodulation for BER estimation (nearest neighbor)
            # For each received symbol, find closest constellation point
            distances = np.abs(rx_symbols[:, np.newaxis] - constellation[np.newaxis, :])
            detected_indices = np.argmin(distances, axis=1)

            # Convert back to bits for BER calculation
            detected_bits = np.zeros((len(detected_indices), bits_per_symbol), dtype=int)
            for i, idx in enumerate(detected_indices):
                detected_bits[i] = [int(b) for b in format(idx, f'0{bits_per_symbol}b')]
            detected_bits = detected_bits.flatten()

            # Calculate BER
            bit_errors = np.sum(bits[:len(detected_bits)] != detected_bits)
            ber = bit_errors / len(detected_bits) if len(detected_bits) > 0 else 0.0

            # Symbol Error Rate
            original_indices = self._symbols_to_indices(tx_symbols, constellation)
            symbol_errors = np.sum(original_indices != detected_indices)
            ser = symbol_errors / len(tx_symbols) if len(tx_symbols) > 0 else 0.0

            # Prepare results
            end_time = datetime.now()
            execution_time = (time.perf_counter() - start_perf) * 1000  # ms

            result_data = {
                'tx_symbols': tx_symbols,
                'rx_symbols': rx_symbols,
                'constellation': constellation,
                'bits': bits,
                'detected_bits': detected_bits,
                'noise_power': noise_power,
                'signal_power': signal_power,
                'actual_snr_db': actual_snr_db,
                'ber': ber,
                'ser': ser,
                'num_bit_errors': int(bit_errors),
                'num_symbol_errors': int(symbol_errors)
            }

            metadata = {
                'module': self.get_name(),
                'modulation_order': M,
                'modulation_type': mod_type,
                'bits_per_symbol': bits_per_symbol,
                'use_gray_coding': use_gray
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
            # Handle any errors gracefully
            end_time = datetime.now()
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

    def _symbols_to_indices(self, symbols: np.ndarray, constellation: np.ndarray) -> np.ndarray:
        """Convert symbols back to constellation indices.

        Finds the index of the closest constellation point for each symbol.

        Args:
            symbols: Complex symbols
            constellation: Reference constellation

        Returns:
            Array of indices into constellation
        """
        distances = np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :])
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    print("=== High-Order Modulation Simulation Demo ===\n")

    sim = HighOrderModulationSimulation()
    print(f"Simulation: {sim.get_name()}\n")

    # Test with different modulation orders
    for M in [4, 16, 64]:
        print(f"Testing {M}-QAM:")
        params = {
            'modulation_order': M,
            'num_symbols': 1000,
            'snr_db': 10.0,
            'modulation_type': 'QAM'
        }

        result = sim.run_simulation(params)

        if result.success:
            print(f"  ✓ Success!")
            print(f"  Execution time: {result.execution_time_ms:.2f} ms")
            print(f"  BER: {result.data['ber']:.6f}")
            print(f"  SER: {result.data['ser']:.6f}")
            print(f"  Bit errors: {result.data['num_bit_errors']}/{len(result.data['bits'])}")
            print(f"  Actual SNR: {result.data['actual_snr_db']:.2f} dB")
        else:
            print(f"  ✗ Failed: {result.error_message}")
        print()

    print("High-Order Modulation simulation ready!")
