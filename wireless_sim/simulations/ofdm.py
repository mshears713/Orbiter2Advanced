"""
OFDM Signal Processing Simulation - Alien Civilization 2

Simulates Orthogonal Frequency Division Multiplexing (OFDM) signal generation,
transmission over AWGN channel, and reception. Demonstrates key OFDM concepts
including subcarrier orthogonality, cyclic prefix, and parallel transmission.

This module represents the second alien civilization's wireless technology,
which uses multiple orthogonal subcarriers for high-speed parallel data transmission.

Educational Goals:
- Understand OFDM signal generation with IFFT/FFT
- Learn the purpose and benefits of cyclic prefix
- Visualize subcarrier constellations
- Explore frequency-domain equalization
- Analyze spectral efficiency of OFDM

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


class OFDMSimulation(SimulationModule):
    """OFDM signal processing simulation with cyclic prefix and AWGN.

    Implements complete OFDM transmitter and receiver chain including:
    - Serial-to-parallel conversion
    - Per-subcarrier modulation (QPSK/QAM)
    - IFFT for time-domain conversion
    - Cyclic prefix insertion
    - AWGN channel
    - Cyclic prefix removal
    - FFT for frequency-domain recovery
    - Per-subcarrier demodulation

    Example:
        >>> sim = OFDMSimulation()
        >>> params = {'num_subcarriers': 64, 'cp_length': 16, 'snr_db': 15.0}
        >>> result = sim.run_simulation(params)
    """

    def get_name(self) -> str:
        return "OFDM Signal Processing (Civilization 2)"

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'num_subcarriers': 64,       # Number of OFDM subcarriers
            'cp_length': 16,              # Cyclic prefix length in samples
            'num_symbols': 10,            # Number of OFDM symbols
            'subcarrier_modulation': 'QPSK',  # Modulation per subcarrier
            'snr_db': 15.0,              # Signal-to-Noise Ratio
            'pilot_spacing': 8           # Spacing between pilot subcarriers
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            'num_subcarriers': {
                'type': 'int',
                'description': 'Number of OFDM subcarriers (FFT size)',
                'min': 16,
                'max': 1024,
                'default': 64,
                'required': True,
                'choices': [16, 32, 64, 128, 256, 512, 1024]
            },
            'cp_length': {
                'type': 'int',
                'description': 'Cyclic prefix length in samples',
                'min': 0,
                'max': 64,
                'default': 16,
                'required': True
            },
            'num_symbols': {
                'type': 'int',
                'description': 'Number of OFDM symbols to transmit',
                'min': 1,
                'max': 100,
                'default': 10,
                'required': True
            },
            'subcarrier_modulation': {
                'type': 'str',
                'description': 'Modulation scheme for each subcarrier',
                'choices': ['BPSK', 'QPSK', '16QAM'],
                'default': 'QPSK',
                'required': False
            },
            'snr_db': {
                'type': 'float',
                'description': 'Signal-to-Noise Ratio in dB',
                'min': -5.0,
                'max': 40.0,
                'default': 15.0,
                'required': True
            },
            'pilot_spacing': {
                'type': 'int',
                'description': 'Spacing between pilot subcarriers (0 for no pilots)',
                'min': 0,
                'max': 32,
                'default': 8,
                'required': False
            }
        }

    def _get_subcarrier_constellation(self, mod_type: str) -> Tuple[np.ndarray, int]:
        """Get constellation and bits per symbol for subcarrier modulation.

        Args:
            mod_type: Modulation type ('BPSK', 'QPSK', '16QAM')

        Returns:
            Tuple of (constellation_points, bits_per_symbol)
        """
        if mod_type == 'BPSK':
            constellation = np.array([-1, 1])
            bits_per_symbol = 1
        elif mod_type == 'QPSK':
            constellation = np.array([
                1+1j, -1+1j, -1-1j, 1-1j
            ]) / np.sqrt(2)
            bits_per_symbol = 2
        elif mod_type == '16QAM':
            # 16-QAM constellation
            levels = np.array([-3, -1, 1, 3])
            I, Q = np.meshgrid(levels, levels)
            constellation = (I + 1j * Q).flatten() / np.sqrt(10)
            bits_per_symbol = 4
        else:
            raise ParameterValidationError(f"Unsupported modulation: {mod_type}")

        return constellation, bits_per_symbol

    def _modulate_subcarriers(
        self,
        bits: np.ndarray,
        num_subcarriers: int,
        constellation: np.ndarray,
        bits_per_symbol: int,
        pilot_spacing: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Modulate bits onto subcarriers.

        Args:
            bits: Binary data stream
            num_subcarriers: Number of OFDM subcarriers
            constellation: Modulation constellation
            bits_per_symbol: Bits per modulation symbol
            pilot_spacing: Spacing for pilot insertion (0 for no pilots)

        Returns:
            Tuple of (modulated_subcarriers, pilot_indices)
        """
        # Determine pilot and data subcarrier indices
        if pilot_spacing > 0:
            pilot_indices = np.arange(0, num_subcarriers, pilot_spacing)
            data_indices = np.setdiff1d(np.arange(num_subcarriers), pilot_indices)
        else:
            pilot_indices = np.array([])
            data_indices = np.arange(num_subcarriers)

        num_data_subcarriers = len(data_indices)

        # Calculate number of complete symbols
        bits_needed = num_data_subcarriers * bits_per_symbol
        num_complete = len(bits) // bits_needed
        bits_to_use = bits[:num_complete * bits_needed]

        # Reshape bits for modulation
        bit_groups = bits_to_use.reshape(-1, bits_per_symbol)

        # Convert bit groups to constellation indices
        powers = 2 ** np.arange(bits_per_symbol)[::-1]
        indices = np.dot(bit_groups, powers)

        # Map to constellation
        data_symbols = constellation[indices]

        # Reshape to symbols x subcarriers
        data_symbols = data_symbols.reshape(num_complete, num_data_subcarriers)

        # Create full subcarrier arrays with pilots
        subcarrier_symbols = np.zeros((num_complete, num_subcarriers), dtype=complex)

        # Insert data symbols
        subcarrier_symbols[:, data_indices] = data_symbols

        # Insert pilot symbols (known reference)
        if len(pilot_indices) > 0:
            subcarrier_symbols[:, pilot_indices] = 1.0  # BPSK pilots

        return subcarrier_symbols, pilot_indices

    def _apply_ifft(self, freq_domain: np.ndarray) -> np.ndarray:
        """Apply IFFT to convert frequency domain to time domain.

        Args:
            freq_domain: Frequency domain symbols (num_symbols x num_subcarriers)

        Returns:
            Time domain signal
        """
        return np.fft.ifft(freq_domain, axis=1)

    def _add_cyclic_prefix(self, time_domain: np.ndarray, cp_length: int) -> np.ndarray:
        """Add cyclic prefix to OFDM symbols.

        The cyclic prefix copies the last cp_length samples to the beginning
        of each OFDM symbol to combat intersymbol interference.

        Args:
            time_domain: Time domain OFDM symbols
            cp_length: Length of cyclic prefix

        Returns:
            OFDM symbols with cyclic prefix
        """
        if cp_length == 0:
            return time_domain

        # Extract last cp_length samples from each symbol
        cp = time_domain[:, -cp_length:]

        # Concatenate CP with symbol
        return np.concatenate([cp, time_domain], axis=1)

    def _remove_cyclic_prefix(self, received: np.ndarray, cp_length: int, fft_size: int) -> np.ndarray:
        """Remove cyclic prefix from received OFDM symbols.

        Args:
            received: Received time domain signal with CP
            cp_length: Length of cyclic prefix
            fft_size: FFT size (number of subcarriers)

        Returns:
            OFDM symbols without CP
        """
        symbol_length = fft_size + cp_length
        num_symbols = len(received) // symbol_length

        # Reshape to symbols
        symbols = received[:num_symbols * symbol_length].reshape(num_symbols, symbol_length)

        # Remove CP
        return symbols[:, cp_length:]

    def _add_awgn(self, signal: np.ndarray, snr_db: float) -> Tuple[np.ndarray, float]:
        """Add AWGN to signal.

        Args:
            signal: Clean signal
            snr_db: Desired SNR in dB

        Returns:
            Tuple of (noisy_signal, noise_power)
        """
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        noise_std = np.sqrt(noise_power / 2)
        noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))

        return signal + noise, noise_power

    def run_simulation(self, parameters: Dict[str, Any]) -> SimulationResult:
        """Execute OFDM simulation.

        Generates OFDM signal, transmits through AWGN channel, and demodulates.

        Args:
            parameters: Simulation parameters

        Returns:
            SimulationResult with OFDM transmission data
        """
        start_time = datetime.now()
        start_perf = time.perf_counter()

        try:
            # Validate parameters
            self.validate_parameters(parameters)

            # Extract parameters
            num_subcarriers = parameters['num_subcarriers']
            cp_length = parameters['cp_length']
            num_ofdm_symbols = parameters['num_symbols']
            mod_type = parameters.get('subcarrier_modulation', 'QPSK')
            snr_db = parameters['snr_db']
            pilot_spacing = parameters.get('pilot_spacing', 8)

            # Get modulation constellation
            constellation, bits_per_symbol = self._get_subcarrier_constellation(mod_type)

            # Determine data subcarriers (accounting for pilots)
            if pilot_spacing > 0:
                num_pilots = num_subcarriers // pilot_spacing
                num_data_subcarriers = num_subcarriers - num_pilots
            else:
                num_data_subcarriers = num_subcarriers

            # Generate random bits
            total_bits = num_ofdm_symbols * num_data_subcarriers * bits_per_symbol
            bits = np.random.randint(0, 2, total_bits)

            # Modulate subcarriers
            freq_domain, pilot_indices = self._modulate_subcarriers(
                bits, num_subcarriers, constellation, bits_per_symbol, pilot_spacing
            )

            # Apply IFFT
            time_domain = self._apply_ifft(freq_domain)

            # Add cyclic prefix
            time_with_cp = self._add_cyclic_prefix(time_domain, cp_length)

            # Flatten to continuous stream
            tx_signal = time_with_cp.flatten()

            # Add AWGN
            rx_signal, noise_power = self._add_awgn(tx_signal, snr_db)

            # Receiver: Remove CP
            rx_time = self._remove_cyclic_prefix(rx_signal, cp_length, num_subcarriers)

            # Apply FFT
            rx_freq = np.fft.fft(rx_time, axis=1)

            # Simple demodulation (nearest neighbor)
            data_indices = np.setdiff1d(np.arange(num_subcarriers), pilot_indices)
            rx_data_symbols = rx_freq[:, data_indices]

            # Demodulate
            distances = np.abs(rx_data_symbols[:, :, np.newaxis] - constellation[np.newaxis, np.newaxis, :])
            detected_indices = np.argmin(distances, axis=2)

            # Convert back to bits
            detected_bits = []
            for idx in detected_indices.flatten():
                bit_str = format(idx, f'0{bits_per_symbol}b')
                detected_bits.extend([int(b) for b in bit_str])
            detected_bits = np.array(detected_bits)

            # Calculate BER
            bit_errors = np.sum(bits[:len(detected_bits)] != detected_bits)
            ber = bit_errors / len(detected_bits) if len(detected_bits) > 0 else 0.0

            # Calculate metrics
            signal_power = np.mean(np.abs(tx_signal)**2)
            actual_snr_db = 10 * np.log10(signal_power / noise_power)

            # Prepare results
            end_time = datetime.now()
            execution_time = (time.perf_counter() - start_perf) * 1000

            result_data = {
                'tx_freq_domain': freq_domain,
                'rx_freq_domain': rx_freq,
                'tx_time_domain': time_domain,
                'rx_time_domain': rx_time,
                'tx_signal': tx_signal,
                'rx_signal': rx_signal,
                'constellation': constellation,
                'pilot_indices': pilot_indices,
                'data_indices': data_indices,
                'bits': bits,
                'detected_bits': detected_bits,
                'ber': ber,
                'num_bit_errors': int(bit_errors),
                'noise_power': noise_power,
                'signal_power': signal_power,
                'actual_snr_db': actual_snr_db
            }

            metadata = {
                'module': self.get_name(),
                'num_subcarriers': num_subcarriers,
                'cp_length': cp_length,
                'num_ofdm_symbols': num_ofdm_symbols,
                'subcarrier_modulation': mod_type,
                'bits_per_symbol': bits_per_symbol,
                'pilot_spacing': pilot_spacing,
                'spectral_efficiency': bits_per_symbol * num_data_subcarriers / (num_subcarriers + cp_length)
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


if __name__ == "__main__":
    print("=== OFDM Signal Processing Simulation Demo ===\n")

    sim = OFDMSimulation()
    print(f"Simulation: {sim.get_name()}\n")

    # Test with different configurations
    configs = [
        {'num_subcarriers': 64, 'cp_length': 16, 'num_symbols': 10, 'snr_db': 15.0},
        {'num_subcarriers': 128, 'cp_length': 32, 'num_symbols': 5, 'snr_db': 20.0}
    ]

    for i, params in enumerate(configs, 1):
        print(f"Configuration {i}:")
        print(f"  Subcarriers: {params['num_subcarriers']}")
        print(f"  CP Length: {params['cp_length']}")
        print(f"  OFDM Symbols: {params['num_symbols']}")

        result = sim.run_simulation(params)

        if result.success:
            print(f"  ✓ Success!")
            print(f"  BER: {result.data['ber']:.6f}")
            print(f"  Bit errors: {result.data['num_bit_errors']}/{len(result.data['bits'])}")
            print(f"  Spectral efficiency: {result.metadata['spectral_efficiency']:.2f} bits/s/Hz")
            print(f"  Execution time: {result.execution_time_ms:.2f} ms")
        else:
            print(f"  ✗ Failed: {result.error_message}")
        print()

    print("OFDM simulation ready!")
