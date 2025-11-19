"""
Base Simulation Module Interface for Orbiter-2

Defines the abstract base class that all simulation modules must inherit from.
This ensures a consistent API across all wireless communication simulations.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
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
