"""
Base Simulation Module Interface for Orbiter-2

Defines the abstract base class that all simulation modules must inherit from.
This ensures a consistent API across all wireless communication simulations.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type
import numpy as np
from models.datamodels import SimulationResult
from simulations.exceptions import SimulationError


class SimulationModule(ABC):
    """Abstract base class for all wireless communication simulation modules.

    All simulation modules in Orbiter-2 must inherit from this class and implement
    its abstract methods. This ensures a consistent interface for parameter handling,
    simulation execution, and result retrieval.

    Subclasses must implement:
        - get_name(): Return simulation module name
        - run_simulation(parameters): Execute the simulation and return results
        - get_default_parameters(): Provide sensible default parameters
        - get_parameter_schema(): Define parameter structure for validation

    Example:
        >>> class MySimulation(SimulationModule):
        ...     def get_name(self) -> str:
        ...         return "MySimulation"
        ...
        ...     def run_simulation(self, parameters: Dict[str, Any]) -> SimulationResult:
        ...         # Implementation here
        ...         pass
    """

    @abstractmethod
    def get_name(self) -> str:
        """Get the human-readable name of this simulation module.

        Returns:
            String name of the simulation (e.g., "High-Order Modulation")

        Example:
            >>> sim = HighOrderModulationSimulation()
            >>> sim.get_name()
            'High-Order Modulation'
        """
        pass

    @abstractmethod
    def run_simulation(self, parameters: Dict[str, Any]) -> SimulationResult:
        """Execute the simulation with given parameters.

        This is the core method that performs the wireless communication simulation.
        It should validate parameters, execute the algorithm, and return results
        in a SimulationResult object.

        Args:
            parameters: Dictionary containing simulation parameters. The exact
                       keys and values depend on the specific simulation type.

        Returns:
            SimulationResult object containing:
                - timestamp: When the simulation was executed
                - parameters: The input parameters used
                - data: Simulation output data (arrays, dicts, etc.)
                - metadata: Additional information about the simulation
                - success: Whether simulation completed successfully
                - error_message: Description of any errors
                - execution_time_ms: How long the simulation took

        Raises:
            SimulationError: If simulation fails due to invalid parameters
                           or execution errors

        Example:
            >>> params = {"snr_db": 10.0, "modulation_order": 16}
            >>> result = sim.run_simulation(params)
            >>> result.success
            True
        """
        pass

    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for this simulation.

        Provides sensible default values for all simulation parameters.
        These defaults should allow the simulation to run successfully
        and produce meaningful educational results.

        Returns:
            Dictionary mapping parameter names to default values

        Example:
            >>> sim.get_default_parameters()
            {'snr_db': 10.0, 'modulation_order': 16, 'num_symbols': 1000}
        """
        pass

    @abstractmethod
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Get the parameter schema for validation.

        Defines the structure, types, and constraints for simulation parameters.
        This schema can be used for automatic validation and UI generation.

        Returns:
            Dictionary describing parameter schema with:
                - parameter_name: {
                    'type': type name (int, float, str, etc.),
                    'description': human-readable description,
                    'min': minimum value (optional),
                    'max': maximum value (optional),
                    'default': default value,
                    'required': whether parameter is required
                  }

        Example:
            >>> sim.get_parameter_schema()
            {
                'snr_db': {
                    'type': 'float',
                    'description': 'Signal-to-Noise Ratio in dB',
                    'min': -10.0,
                    'max': 40.0,
                    'default': 10.0,
                    'required': True
                },
                'modulation_order': {
                    'type': 'int',
                    'description': 'QAM modulation order (power of 2)',
                    'min': 2,
                    'max': 256,
                    'default': 16,
                    'required': True
                }
            }
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate simulation parameters against the schema.

        Checks that all required parameters are present and that values
        meet the constraints defined in the parameter schema.

        Args:
            parameters: Dictionary of parameters to validate

        Returns:
            True if parameters are valid

        Raises:
            SimulationError: If parameters are invalid, with detailed message

        Example:
            >>> sim.validate_parameters({'snr_db': 10.0})
            SimulationError: Missing required parameter: modulation_order
        """
        schema = self.get_parameter_schema()

        # Check required parameters
        for param_name, param_spec in schema.items():
            if param_spec.get('required', False) and param_name not in parameters:
                raise SimulationError(f"Missing required parameter: {param_name}")

        # Validate parameter values
        for param_name, param_value in parameters.items():
            if param_name not in schema:
                # Allow extra parameters (for future extensibility)
                continue

            param_spec = schema[param_name]

            # Check type (basic check)
            expected_type = param_spec.get('type')
            if expected_type == 'int' and not isinstance(param_value, int):
                raise SimulationError(
                    f"Parameter '{param_name}' must be int, got {type(param_value).__name__}"
                )
            elif expected_type == 'float' and not isinstance(param_value, (int, float)):
                raise SimulationError(
                    f"Parameter '{param_name}' must be float, got {type(param_value).__name__}"
                )

            # Check min/max bounds
            if 'min' in param_spec and param_value < param_spec['min']:
                raise SimulationError(
                    f"Parameter '{param_name}' must be >= {param_spec['min']}, got {param_value}"
                )
            if 'max' in param_spec and param_value > param_spec['max']:
                raise SimulationError(
                    f"Parameter '{param_name}' must be <= {param_spec['max']}, got {param_value}"
                )

        return True

    def get_description(self) -> str:
        """Get a detailed description of this simulation module.

        Returns:
            Multi-line string describing what this simulation does

        Note:
            Subclasses can override this to provide custom descriptions
        """
        return f"{self.get_name()} simulation module"

    def merge_with_defaults(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Merge provided parameters with defaults.

        Fills in any missing parameters with default values from
        get_default_parameters().

        Args:
            parameters: User-provided parameters (may be partial)

        Returns:
            Complete parameter dictionary with defaults filled in

        Example:
            >>> params = {'snr_db': 15.0}
            >>> complete = sim.merge_with_defaults(params)
            >>> complete
            {'snr_db': 15.0, 'modulation_order': 16, 'num_symbols': 1000}
        """
        defaults = self.get_default_parameters()
        merged = defaults.copy()
        merged.update(parameters)
        return merged

    def add_awgn_noise(
        self,
        signal: np.ndarray,
        snr_db: float,
        signal_power: float = None
    ) -> tuple[np.ndarray, float]:
        """Add Additive White Gaussian Noise to signal.

        Helper method for adding AWGN to achieve a specified SNR.
        Handles both real and complex signals.

        Args:
            signal: Input signal (real or complex numpy array)
            snr_db: Desired SNR in decibels
            signal_power: Pre-computed signal power (optional, calculated if None)

        Returns:
            Tuple of (noisy_signal, noise_power_used)

        Example:
            >>> clean_signal = np.array([1+1j, -1+1j, -1-1j, 1-1j])
            >>> noisy, noise_pwr = sim.add_awgn_noise(clean_signal, 10.0)
        """
        # Calculate signal power if not provided
        if signal_power is None:
            signal_power = np.mean(np.abs(signal)**2)

        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10.0)

        # Calculate required noise power
        noise_power = signal_power / snr_linear

        # Generate noise based on signal type
        if np.iscomplexobj(signal):
            # Complex signal: split noise between I and Q
            noise_std = np.sqrt(noise_power / 2)
            noise = noise_std * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
        else:
            # Real signal
            noise_std = np.sqrt(noise_power)
            noise = noise_std * np.random.randn(*signal.shape)

        noisy_signal = signal + noise
        return noisy_signal, noise_power

    def calculate_ber(
        self,
        transmitted_bits: np.ndarray,
        received_bits: np.ndarray
    ) -> tuple[float, int]:
        """Calculate Bit Error Rate.

        Compares transmitted and received bit sequences to compute BER.

        Args:
            transmitted_bits: Original transmitted bits (0s and 1s)
            received_bits: Decoded received bits (0s and 1s)

        Returns:
            Tuple of (ber, num_errors)
                - ber: Bit error rate (fraction)
                - num_errors: Number of bit errors

        Example:
            >>> tx_bits = np.array([1, 0, 1, 1, 0])
            >>> rx_bits = np.array([1, 0, 0, 1, 0])
            >>> ber, errors = sim.calculate_ber(tx_bits, rx_bits)
            >>> ber
            0.2
            >>> errors
            1
        """
        # Ensure equal length (truncate to shorter if needed)
        min_len = min(len(transmitted_bits), len(received_bits))
        tx = transmitted_bits[:min_len]
        rx = received_bits[:min_len]

        # Count errors
        errors = np.sum(tx != rx)
        ber = errors / min_len if min_len > 0 else 0.0

        return ber, int(errors)

    def calculate_ser(
        self,
        transmitted_symbols: np.ndarray,
        received_symbols: np.ndarray,
        constellation: np.ndarray = None
    ) -> tuple[float, int]:
        """Calculate Symbol Error Rate.

        Compares transmitted and received symbol sequences to compute SER.
        If constellation is provided, performs nearest-neighbor detection
        on received symbols first.

        Args:
            transmitted_symbols: Original transmitted symbols
            received_symbols: Received symbols (possibly noisy)
            constellation: Reference constellation for detection (optional)

        Returns:
            Tuple of (ser, num_errors)
                - ser: Symbol error rate (fraction)
                - num_errors: Number of symbol errors

        Example:
            >>> tx_syms = np.array([1+1j, -1+1j, 1-1j])
            >>> rx_syms = np.array([0.9+1.1j, -1.1+0.9j, 1.1-1.2j])
            >>> qpsk = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            >>> ser, errors = sim.calculate_ser(tx_syms, rx_syms, qpsk)
        """
        min_len = min(len(transmitted_symbols), len(received_symbols))
        tx = transmitted_symbols[:min_len]
        rx = received_symbols[:min_len]

        if constellation is not None:
            # Perform nearest-neighbor detection
            distances = np.abs(rx[:, np.newaxis] - constellation[np.newaxis, :])
            detected_indices = np.argmin(distances, axis=1)
            detected_symbols = constellation[detected_indices]

            # Compare detected symbols to transmitted
            errors = np.sum(~np.isclose(tx, detected_symbols, rtol=1e-5))
        else:
            # Direct comparison (symbols must match exactly)
            errors = np.sum(~np.isclose(tx, rx, rtol=1e-5))

        ser = errors / min_len if min_len > 0 else 0.0
        return ser, int(errors)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get module capabilities and metadata.

        Returns information about what this simulation can do,
        useful for dynamic UI generation and module discovery.

        Returns:
            Dictionary with:
                - name: Module name
                - parameters: Parameter schema
                - defaults: Default parameter values
                - description: Module description
                - tags: List of relevant tags

        Example:
            >>> caps = sim.get_capabilities()
            >>> caps['name']
            'High-Order Modulation'
        """
        return {
            'name': self.get_name(),
            'description': self.get_description(),
            'parameters': self.get_parameter_schema(),
            'defaults': self.get_default_parameters(),
            'tags': self.get_tags() if hasattr(self, 'get_tags') else []
        }


# Module Registry for dynamic discovery
_simulation_registry: List[Type[SimulationModule]] = []


def register_simulation(cls: Type[SimulationModule]) -> Type[SimulationModule]:
    """Decorator to register a simulation module.

    Use this decorator on simulation classes to automatically
    register them for discovery.

    Args:
        cls: SimulationModule subclass to register

    Returns:
        The same class (allows use as decorator)

    Example:
        >>> @register_simulation
        ... class MySimulation(SimulationModule):
        ...     pass
    """
    if cls not in _simulation_registry:
        _simulation_registry.append(cls)
    return cls


def get_registered_simulations() -> List[Type[SimulationModule]]:
    """Get all registered simulation modules.

    Returns:
        List of registered SimulationModule classes

    Example:
        >>> sims = get_registered_simulations()
        >>> for sim_class in sims:
        ...     print(sim_class().get_name())
    """
    return _simulation_registry.copy()


def discover_simulations() -> Dict[str, Type[SimulationModule]]:
    """Discover all available simulation modules.

    Returns:
        Dictionary mapping simulation names to their classes

    Example:
        >>> available = discover_simulations()
        >>> available.keys()
        dict_keys(['High-Order Modulation', 'OFDM Signal Processing', ...])
    """
    return {sim_class().get_name(): sim_class for sim_class in _simulation_registry}


if __name__ == "__main__":
    # This module defines interfaces only, no standalone demo
    print("=== Orbiter-2 Simulation Module Base Class ===\n")
    print("This module defines the abstract base class for all simulations.")
    print("Subclasses must implement:")
    print("  - get_name()")
    print("  - run_simulation(parameters)")
    print("  - get_default_parameters()")
    print("  - get_parameter_schema()")
    print("\nSee individual simulation modules for concrete implementations.")
