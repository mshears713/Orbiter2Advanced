"""
Interference Shaping Simulation - Alien Civilization 6

Simulates spatial interference control using beamforming and null steering.
Demonstrates how multiple antennas can be used to direct signal energy toward
desired users while minimizing interference to others.

This represents advanced MIMO and beamforming techniques used in 5G, WiFi 6,
and modern radar systems.

Educational Goals:
- Understand beamforming and spatial filtering
- Learn array factor and radiation patterns
- Visualize spatial interference distribution
- Explore null steering for interference mitigation
- Analyze SINR (Signal-to-Interference-plus-Noise Ratio)

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
import time

from simulations.base import SimulationModule
from simulations.exceptions import SimulationError
from models.datamodels import SimulationResult


class InterferenceShapingSimulation(SimulationModule):
    """Spatial interference shaping with antenna arrays.

    Implements beamforming with configurable antenna arrays to
    shape the spatial radiation pattern and control interference.

    Example:
        >>> sim = InterferenceShapingSimulation()
        >>> params = {'num_antennas': 8, 'target_angle': 30, 'null_angles': [60, -45]}
        >>> result = sim.run_simulation(params)
    """

    def get_name(self) -> str:
        return "Interference Shaping (Civilization 6)"

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'num_antennas': 8,          # Number of array elements
            'antenna_spacing': 0.5,     # Spacing in wavelengths
            'target_angle': 30.0,       # Desired signal direction (degrees)
            'interferer_angles': [-45.0, 60.0],  # Interferer directions
            'snr_db': 10.0,             # Signal-to-Noise Ratio
            'inr_db': 20.0              # Interference-to-Noise Ratio
        }

    def get_parameter_schema(self) -> Dict[str, Any]:
        return {
            'num_antennas': {
                'type': 'int',
                'description': 'Number of antenna elements in array',
                'min': 2,
                'max': 16,
                'default': 8,
                'required': True
            },
            'antenna_spacing': {
                'type': 'float',
                'description': 'Antenna spacing in wavelengths',
                'min': 0.25,
                'max': 1.0,
                'default': 0.5,
                'required': True
            },
            'target_angle': {
                'type': 'float',
                'description': 'Target signal angle in degrees',
                'min': -90.0,
                'max': 90.0,
                'default': 30.0,
                'required': True
            },
            'snr_db': {
                'type': 'float',
                'description': 'Signal-to-Noise Ratio in dB',
                'min': 0.0,
                'max': 30.0,
                'default': 10.0,
                'required': True
            }
        }

    def _compute_array_response(
        self,
        num_antennas: int,
        spacing: float,
        angles_deg: np.ndarray
    ) -> np.ndarray:
        """Compute array response vector for given angles.

        Args:
            num_antennas: Number of array elements
            spacing: Element spacing in wavelengths
            angles_deg: Angles to compute response for (degrees)

        Returns:
            Array response matrix (num_angles x num_antennas)
        """
        angles_rad = np.deg2rad(angles_deg)

        # Array element positions (uniform linear array)
        element_positions = np.arange(num_antennas) * spacing

        # Compute phase shifts
        # For angle theta, phase at element n is: 2*pi*d*sin(theta)
        # where d is position in wavelengths
        phase_matrix = 2 * np.pi * np.outer(np.sin(angles_rad), element_positions)

        # Array response is complex exponential of phase
        array_response = np.exp(1j * phase_matrix)

        return array_response

    def _design_beamformer_weights(
        self,
        array_response_target: np.ndarray,
        array_response_interferers: np.ndarray = None,
        method: str = 'max_gain'
    ) -> np.ndarray:
        """Design beamformer weights.

        Args:
            array_response_target: Array response for target direction
            array_response_interferers: Array responses for interferers (optional)
            method: Beamforming method ('max_gain' or 'null_steering')

        Returns:
            Complex beamformer weight vector
        """
        num_antennas = len(array_response_target)

        if method == 'max_gain' or array_response_interferers is None:
            # Simple maximum ratio combining (MRC)
            # Weights = conjugate of array response (matched filter)
            weights = np.conj(array_response_target) / num_antennas
        else:
            # Null steering: place nulls at interferer directions
            # Use MVDR (Minimum Variance Distortionless Response)

            # Constraint matrix: maintain unit gain at target
            C = array_response_target.reshape(-1, 1)

            # Interference covariance (simplified)
            R = np.eye(num_antennas)  # Start with identity

            # Add interferer contributions
            for interferer_response in array_response_interferers.T:
                R += 10 * np.outer(interferer_response, np.conj(interferer_response))

            # MVDR weights: w = R^{-1} * a / (a^H * R^{-1} * a)
            try:
                R_inv = np.linalg.inv(R)
                numerator = R_inv @ array_response_target
                denominator = np.conj(array_response_target) @ numerator
                weights = numerator / denominator
            except:
                # Fallback to MRC if inversion fails
                weights = np.conj(array_response_target) / num_antennas

        return weights

    def _compute_array_pattern(
        self,
        weights: np.ndarray,
        num_antennas: int,
        spacing: float,
        angles_deg: np.ndarray
    ) -> np.ndarray:
        """Compute array pattern (beampattern) for given weights.

        Args:
            weights: Beamformer weight vector
            num_antennas: Number of antennas
            spacing: Antenna spacing
            angles_deg: Angles to evaluate pattern at

        Returns:
            Array pattern (complex gain vs angle)
        """
        # Get array response for all angles
        array_response = self._compute_array_response(num_antennas, spacing, angles_deg)

        # Apply weights: pattern = w^H * a(theta)
        pattern = array_response @ np.conj(weights)

        return pattern

    def run_simulation(self, parameters: Dict[str, Any]) -> SimulationResult:
        """Execute interference shaping simulation.

        Args:
            parameters: Simulation parameters

        Returns:
            SimulationResult with beamforming data
        """
        start_time = datetime.now()
        start_perf = time.perf_counter()

        try:
            # Validate
            self.validate_parameters(parameters)

            # Extract parameters
            num_antennas = parameters['num_antennas']
            spacing = parameters['antenna_spacing']
            target_angle = parameters['target_angle']
            interferer_angles = parameters.get('interferer_angles', [-45.0, 60.0])
            snr_db = parameters['snr_db']
            inr_db = parameters.get('inr_db', 20.0)

            # Compute array responses
            target_response = self._compute_array_response(
                num_antennas, spacing, np.array([target_angle])
            )[0]

            interferer_responses = self._compute_array_response(
                num_antennas, spacing, np.array(interferer_angles)
            )

            # Design beamformer weights (null steering)
            weights_null_steering = self._design_beamformer_weights(
                target_response,
                interferer_responses,
                method='null_steering'
            )

            # Design simple MRC weights for comparison
            weights_mrc = self._design_beamformer_weights(
                target_response,
                None,
                method='max_gain'
            )

            # Compute array patterns
            angle_grid = np.linspace(-90, 90, 361)

            pattern_null = self._compute_array_pattern(
                weights_null_steering, num_antennas, spacing, angle_grid
            )

            pattern_mrc = self._compute_array_pattern(
                weights_mrc, num_antennas, spacing, angle_grid
            )

            # Calculate SINR improvement
            # Signal gain at target
            signal_gain_null = np.abs(pattern_null[np.argmin(np.abs(angle_grid - target_angle))])**2
            signal_gain_mrc = np.abs(pattern_mrc[np.argmin(np.abs(angle_grid - target_angle))])**2

            # Interference suppression
            interference_gains_null = []
            interference_gains_mrc = []

            for int_angle in interferer_angles:
                idx = np.argmin(np.abs(angle_grid - int_angle))
                interference_gains_null.append(np.abs(pattern_null[idx])**2)
                interference_gains_mrc.append(np.abs(pattern_mrc[idx])**2)

            avg_int_suppression_null = np.mean(interference_gains_null)
            avg_int_suppression_mrc = np.mean(interference_gains_mrc)

            # SINR calculation (simplified)
            sinr_null_steering_db = 10 * np.log10(signal_gain_null / (avg_int_suppression_null + 0.01))
            sinr_mrc_db = 10 * np.log10(signal_gain_mrc / (avg_int_suppression_mrc + 0.01))

            sinr_improvement_db = sinr_null_steering_db - sinr_mrc_db

            # Execution time
            execution_time = (time.perf_counter() - start_perf) * 1000

            result_data = {
                'weights_null_steering': weights_null_steering,
                'weights_mrc': weights_mrc,
                'angle_grid': angle_grid,
                'pattern_null_steering': pattern_null,
                'pattern_mrc': pattern_mrc,
                'target_angle': target_angle,
                'interferer_angles': interferer_angles,
                'signal_gain_null': float(signal_gain_null),
                'signal_gain_mrc': float(signal_gain_mrc),
                'sinr_null_db': float(sinr_null_steering_db),
                'sinr_mrc_db': float(sinr_mrc_db),
                'sinr_improvement_db': float(sinr_improvement_db)
            }

            metadata = {
                'module': self.get_name(),
                'num_antennas': num_antennas,
                'antenna_spacing': spacing,
                'num_interferers': len(interferer_angles)
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
    print("=== Interference Shaping Simulation Demo ===\n")

    sim = InterferenceShapingSimulation()
    print(f"Simulation: {sim.get_name()}\n")

    params = {
        'num_antennas': 8,
        'antenna_spacing': 0.5,
        'target_angle': 30.0,
        'interferer_angles': [-45.0, 60.0],
        'snr_db': 10.0
    }

    result = sim.run_simulation(params)

    if result.success:
        print("Beamforming Results:")
        print(f"  Target Angle: {result.data['target_angle']}°")
        print(f"  Interferer Angles: {result.data['interferer_angles']}")
        print(f"  SINR (Null Steering): {result.data['sinr_null_db']:.2f} dB")
        print(f"  SINR (MRC): {result.data['sinr_mrc_db']:.2f} dB")
        print(f"  SINR Improvement: {result.data['sinr_improvement_db']:.2f} dB")
        print()
    else:
        print(f"Failed: {result.error_message}\n")

    print("Interference shaping simulation ready!")
