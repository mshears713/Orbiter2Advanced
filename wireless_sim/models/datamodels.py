"""
Core Data Models for Orbiter-2 Wireless Communications Simulation

Defines type-safe dataclasses for domain entities used throughout the simulation
framework. All models use type hints, validation, and provide serialization support.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import json


@dataclass
class SignalParameters:
    """Parameters defining a wireless signal for simulation.

    This class encapsulates all key parameters needed to generate and process
    wireless communication signals across different modulation schemes.

    Attributes:
        modulation_order: Constellation size (e.g., 16 for 16-QAM). Must be power of 2.
        bandwidth: Signal bandwidth in Hz (e.g., 20e6 for 20 MHz)
        snr_db: Signal-to-Noise Ratio in decibels (e.g., 10.0)
        num_symbols: Number of symbols to generate (e.g., 1000)
        sampling_rate: Sampling rate in Hz (e.g., 1e6)
        carrier_frequency: Carrier frequency in Hz (e.g., 2.4e9 for 2.4 GHz)

    Raises:
        ValueError: If modulation_order is not a power of 2 or if any value is negative.

    Example:
        >>> params = SignalParameters(modulation_order=16, bandwidth=20e6, snr_db=10.0)
        >>> params.bits_per_symbol
        4
    """
    modulation_order: int
    bandwidth: float
    snr_db: float
    num_symbols: int = 1000
    sampling_rate: float = 1e6
    carrier_frequency: float = 2.4e9

    def __post_init__(self):
        """Validate parameters after initialization."""
        # Validate modulation order is a power of 2
        if self.modulation_order <= 0 or (self.modulation_order & (self.modulation_order - 1)) != 0:
            raise ValueError(
                f"modulation_order must be a power of 2, got {self.modulation_order}"
            )

        # Validate positive values
        if self.bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {self.bandwidth}")
        if self.num_symbols <= 0:
            raise ValueError(f"num_symbols must be positive, got {self.num_symbols}")
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {self.sampling_rate}")
        if self.carrier_frequency < 0:
            raise ValueError(f"carrier_frequency must be non-negative, got {self.carrier_frequency}")

    @property
    def bits_per_symbol(self) -> int:
        """Calculate bits per symbol from modulation order.

        Returns:
            Number of bits encoded in each symbol (log2 of modulation order).
        """
        return int(np.log2(self.modulation_order))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ChannelDescription:
    """Description of a wireless communication channel.

    Models channel characteristics including fading, multipath, and other impairments
    that affect signal propagation.

    Attributes:
        channel_type: Type of channel (e.g., "AWGN", "Rayleigh", "Rician")
        fading: Whether fading is enabled
        multipath_taps: Number of multipath taps for dispersive channels
        doppler_shift: Maximum Doppler shift in Hz (for mobile scenarios)
        usage_notes: Human-readable description of channel characteristics

    Example:
        >>> channel = ChannelDescription(
        ...     channel_type="Rayleigh",
        ...     fading=True,
        ...     multipath_taps=3,
        ...     usage_notes="Mobile urban environment"
        ... )
    """
    channel_type: str
    fading: bool = False
    multipath_taps: int = 1
    doppler_shift: float = 0.0
    usage_notes: str = ""

    def __post_init__(self):
        """Validate channel parameters."""
        if self.multipath_taps < 1:
            raise ValueError(f"multipath_taps must be at least 1, got {self.multipath_taps}")
        if self.doppler_shift < 0:
            raise ValueError(f"doppler_shift must be non-negative, got {self.doppler_shift}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SimulationResult:
    """Container for simulation outputs and metadata.

    Stores all results from a simulation run including the processed data,
    input parameters, timestamps, and success indicators. Supports serialization
    for storage and visualization.

    Attributes:
        timestamp: When the simulation was executed
        parameters: Input parameters used for the simulation
        data: Primary simulation output data (numpy array or dict)
        metadata: Additional information about the simulation
        success: Whether the simulation completed successfully
        error_message: Error description if simulation failed
        execution_time_ms: Simulation execution time in milliseconds

    Example:
        >>> import numpy as np
        >>> result = SimulationResult(
        ...     timestamp=datetime.now(),
        ...     parameters={"snr_db": 10.0},
        ...     data=np.array([1, 2, 3])
        ... )
    """
    timestamp: datetime
    parameters: Dict[str, Any]
    data: Any  # Typically np.ndarray or dict of arrays
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Note: numpy arrays are converted to lists for JSON serialization.
        """
        result_dict = {
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'metadata': self.metadata,
            'success': self.success,
            'error_message': self.error_message,
            'execution_time_ms': self.execution_time_ms
        }

        # Handle numpy array serialization
        if isinstance(self.data, np.ndarray):
            result_dict['data'] = self.data.tolist()
            result_dict['data_shape'] = self.data.shape
            result_dict['data_dtype'] = str(self.data.dtype)
        elif isinstance(self.data, dict):
            result_dict['data'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.data.items()
            }
        else:
            result_dict['data'] = self.data

        return result_dict

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class ConstellationPoint:
    """Represents a point in a modulation constellation.

    Used for visualization and analysis of modulation schemes.

    Attributes:
        real: Real (I) component
        imag: Imaginary (Q) component
        symbol_index: Index of the symbol this point represents
        label: Optional label for the point

    Example:
        >>> point = ConstellationPoint(real=1.0, imag=1.0, symbol_index=15)
        >>> point.magnitude
        1.4142135623730951
    """
    real: float
    imag: float
    symbol_index: int
    label: str = ""

    @property
    def magnitude(self) -> float:
        """Calculate magnitude of complex point."""
        return np.sqrt(self.real**2 + self.imag**2)

    @property
    def phase(self) -> float:
        """Calculate phase angle in radians."""
        return np.arctan2(self.imag, self.real)

    @property
    def complex_value(self) -> complex:
        """Return as complex number."""
        return complex(self.real, self.imag)


@dataclass
class NetworkNode:
    """Represents a node in a wireless network.

    Used for mesh routing and network topology simulations.

    Attributes:
        node_id: Unique identifier for the node
        x_position: X coordinate in meters
        y_position: Y coordinate in meters
        transmission_power: Transmission power in dBm
        transmission_range: Maximum transmission range in meters
        neighbors: List of neighboring node IDs

    Example:
        >>> node = NetworkNode(node_id=1, x_position=100.0, y_position=50.0)
        >>> node.distance_to(200.0, 150.0)
        141.4213562373095
    """
    node_id: int
    x_position: float
    y_position: float
    transmission_power: float = 20.0  # dBm
    transmission_range: float = 100.0  # meters
    neighbors: List[int] = field(default_factory=list)

    def distance_to(self, x: float, y: float) -> float:
        """Calculate Euclidean distance to another point.

        Args:
            x: X coordinate of target point
            y: Y coordinate of target point

        Returns:
            Distance in meters
        """
        return np.sqrt((self.x_position - x)**2 + (self.y_position - y)**2)

    def can_reach(self, other_node: 'NetworkNode') -> bool:
        """Determine if this node can reach another node.

        Args:
            other_node: Target node to check reachability

        Returns:
            True if target node is within transmission range
        """
        distance = self.distance_to(other_node.x_position, other_node.y_position)
        return distance <= self.transmission_range

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class Packet:
    """Represents a data packet in network simulations.

    Attributes:
        packet_id: Unique packet identifier
        source_id: Source node ID
        destination_id: Destination node ID
        payload_size: Payload size in bytes
        creation_time: When packet was created
        delivery_time: When packet was delivered (None if not delivered)
        hop_count: Number of hops taken
        route: List of node IDs in the packet's route

    Example:
        >>> packet = Packet(packet_id=1, source_id=0, destination_id=5, payload_size=1024)
    """
    packet_id: int
    source_id: int
    destination_id: int
    payload_size: int
    creation_time: float = 0.0
    delivery_time: Optional[float] = None
    hop_count: int = 0
    route: List[int] = field(default_factory=list)

    @property
    def is_delivered(self) -> bool:
        """Check if packet has been delivered."""
        return self.delivery_time is not None

    @property
    def delay(self) -> Optional[float]:
        """Calculate packet delay if delivered.

        Returns:
            Delay in seconds, or None if not delivered
        """
        if self.delivery_time is not None:
            return self.delivery_time - self.creation_time
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CodedBits:
    """Container for channel-coded bitstreams.

    Attributes:
        uncoded_bits: Original information bits
        coded_bits: Encoded bits after channel coding
        code_rate: Code rate (ratio of uncoded to coded bits)
        coding_scheme: Name of coding scheme used

    Example:
        >>> coded = CodedBits(
        ...     uncoded_bits=np.array([1, 0, 1, 1]),
        ...     coded_bits=np.array([1, 1, 0, 1, 1, 0, 0, 1]),
        ...     code_rate=0.5,
        ...     coding_scheme="Convolutional"
        ... )
    """
    uncoded_bits: np.ndarray
    coded_bits: np.ndarray
    code_rate: float
    coding_scheme: str

    def __post_init__(self):
        """Validate coded bits."""
        if self.code_rate <= 0 or self.code_rate > 1:
            raise ValueError(f"code_rate must be in (0, 1], got {self.code_rate}")

    @property
    def coding_overhead(self) -> float:
        """Calculate coding overhead ratio."""
        return len(self.coded_bits) / len(self.uncoded_bits)


# Utility functions for working with data models

def create_awgn_channel(snr_db: float, notes: str = "") -> ChannelDescription:
    """Create an AWGN channel description.

    Args:
        snr_db: Target SNR in dB
        notes: Optional usage notes

    Returns:
        ChannelDescription for AWGN channel
    """
    return ChannelDescription(
        channel_type="AWGN",
        fading=False,
        multipath_taps=1,
        doppler_shift=0.0,
        usage_notes=notes or f"AWGN channel with {snr_db} dB SNR"
    )


def create_rayleigh_channel(doppler_hz: float = 100.0, taps: int = 3) -> ChannelDescription:
    """Create a Rayleigh fading channel description.

    Args:
        doppler_hz: Maximum Doppler shift in Hz
        taps: Number of multipath taps

    Returns:
        ChannelDescription for Rayleigh fading channel
    """
    return ChannelDescription(
        channel_type="Rayleigh",
        fading=True,
        multipath_taps=taps,
        doppler_shift=doppler_hz,
        usage_notes=f"Rayleigh fading with {doppler_hz} Hz Doppler, {taps} taps"
    )


if __name__ == "__main__":
    # Demo and test data models
    print("=== Orbiter-2 Data Models Demo ===\n")

    # Test SignalParameters
    print("1. SignalParameters:")
    sig_params = SignalParameters(
        modulation_order=64,
        bandwidth=20e6,
        snr_db=15.0,
        num_symbols=500
    )
    print(f"   Modulation: {sig_params.modulation_order}-QAM")
    print(f"   Bits per symbol: {sig_params.bits_per_symbol}")
    print(f"   SNR: {sig_params.snr_db} dB\n")

    # Test ChannelDescription
    print("2. ChannelDescription:")
    channel = create_rayleigh_channel(doppler_hz=150.0, taps=4)
    print(f"   Type: {channel.channel_type}")
    print(f"   Fading: {channel.fading}")
    print(f"   Notes: {channel.usage_notes}\n")

    # Test SimulationResult
    print("3. SimulationResult:")
    result = SimulationResult(
        timestamp=datetime.now(),
        parameters={"snr_db": 10.0, "mod_order": 16},
        data=np.array([1.0 + 1.0j, -1.0 + 1.0j, 1.0 - 1.0j]),
        metadata={"module": "HighOrderModulation"},
        execution_time_ms=125.5
    )
    print(f"   Success: {result.success}")
    print(f"   Execution time: {result.execution_time_ms} ms")
    print(f"   Data shape: {result.data.shape}\n")

    # Test NetworkNode
    print("4. NetworkNode:")
    node1 = NetworkNode(node_id=1, x_position=0.0, y_position=0.0)
    node2 = NetworkNode(node_id=2, x_position=75.0, y_position=0.0)
    print(f"   Node 1 -> Node 2 distance: {node1.distance_to(node2.x_position, node2.y_position):.1f} m")
    print(f"   Can reach: {node1.can_reach(node2)}\n")

    # Test Packet
    print("5. Packet:")
    packet = Packet(
        packet_id=42,
        source_id=1,
        destination_id=5,
        payload_size=1024,
        creation_time=0.0
    )
    packet.delivery_time = 0.150  # 150 ms
    print(f"   Packet ID: {packet.packet_id}")
    print(f"   Delivered: {packet.is_delivered}")
    print(f"   Delay: {packet.delay * 1000:.1f} ms\n")

    print("All data models validated successfully!")
