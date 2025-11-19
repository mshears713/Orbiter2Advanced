"""
Configuration Module for Orbiter-2 Wireless Communications Simulation

This module centralizes all configuration parameters for simulations, visualizations,
database, and UI settings. Supports loading from and saving to JSON configuration files
with robust error handling and fallback to sensible defaults.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class SimulationDefaults:
    """Default parameters for wireless communication simulations.

    Attributes:
        sampling_rate: Signal sampling rate in Hz (default: 1e6 = 1 MHz)
        symbol_rate: Symbol transmission rate in symbols/second (default: 100e3 = 100 kHz)
        num_symbols: Default number of symbols to generate per simulation (default: 1000)
        modulation_order: Default QAM/PSK modulation order - must be power of 2 (default: 16)
        snr_db: Signal-to-Noise Ratio in decibels (default: 10.0 dB)
        carrier_frequency: Carrier frequency in Hz (default: 2.4e9 = 2.4 GHz)
    """
    sampling_rate: float = 1e6
    symbol_rate: float = 100e3
    num_symbols: int = 1000
    modulation_order: int = 16
    snr_db: float = 10.0
    carrier_frequency: float = 2.4e9


@dataclass
class OFDMDefaults:
    """OFDM-specific default parameters.

    Attributes:
        num_subcarriers: Number of orthogonal subcarriers (default: 64)
        cyclic_prefix_length: Length of cyclic prefix in samples (default: 16)
        subcarrier_spacing: Frequency spacing between subcarriers in Hz (default: 15e3)
    """
    num_subcarriers: int = 64
    cyclic_prefix_length: int = 16
    subcarrier_spacing: float = 15e3


@dataclass
class CodingDefaults:
    """Channel coding default parameters.

    Attributes:
        constraint_length: Convolutional code constraint length (default: 7)
        code_rate: Code rate as fraction (default: 0.5 for rate 1/2)
        ldpc_block_length: LDPC code block length (default: 1024)
        ldpc_max_iterations: Maximum LDPC decoding iterations (default: 50)
    """
    constraint_length: int = 7
    code_rate: float = 0.5
    ldpc_block_length: int = 1024
    ldpc_max_iterations: int = 50


@dataclass
class NetworkDefaults:
    """Network simulation default parameters.

    Attributes:
        num_nodes: Number of network nodes (default: 10)
        transmission_range: Node transmission range in meters (default: 100.0)
        packet_size: Packet size in bytes (default: 1024)
    """
    num_nodes: int = 10
    transmission_range: float = 100.0
    packet_size: int = 1024


@dataclass
class DatabaseConfig:
    """Database configuration parameters.

    Attributes:
        db_path: Path to SQLite database file
        enable_logging: Enable database logging (default: True)
    """
    db_path: str = "wireless_sim.db"
    enable_logging: bool = True


@dataclass
class VisualizationConfig:
    """Visualization settings.

    Attributes:
        dpi: Matplotlib figure DPI (default: 100)
        figure_width: Figure width in inches (default: 10)
        figure_height: Figure height in inches (default: 6)
        animation_interval: Animation frame interval in milliseconds (default: 100)
        theme: Plotly theme (default: "plotly")
    """
    dpi: int = 100
    figure_width: int = 10
    figure_height: int = 6
    animation_interval: int = 100
    theme: str = "plotly"


@dataclass
class Config:
    """Master configuration class aggregating all settings.

    This class brings together all configuration domains (simulations, database,
    visualizations) into a single cohesive configuration object.

    Attributes:
        simulation: Simulation default parameters
        ofdm: OFDM-specific parameters
        coding: Channel coding parameters
        network: Network simulation parameters
        database: Database configuration
        visualization: Visualization settings
    """
    simulation: SimulationDefaults = field(default_factory=SimulationDefaults)
    ofdm: OFDMDefaults = field(default_factory=OFDMDefaults)
    coding: CodingDefaults = field(default_factory=CodingDefaults)
    network: NetworkDefaults = field(default_factory=NetworkDefaults)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from JSON file with fallback to defaults.

    Attempts to load configuration from the specified JSON file. If the file
    doesn't exist, is malformed, or any errors occur, falls back to default
    configuration with appropriate error logging.

    Args:
        config_path: Path to JSON configuration file. If None, uses default
                     'config.json' in the config directory.

    Returns:
        Config object populated from file or with default values.

    Raises:
        No exceptions raised - errors are logged and defaults are used.

    Example:
        >>> config = load_config('my_config.json')
        >>> print(config.simulation.snr_db)
        10.0
    """
    # Determine config file path
    if config_path is None:
        config_dir = Path(__file__).parent
        config_path = config_dir / "config.json"
    else:
        config_path = Path(config_path)

    # If file doesn't exist, return defaults
    if not config_path.exists():
        print(f"Configuration file not found at {config_path}. Using defaults.")
        return Config()

    try:
        # Load JSON configuration
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Reconstruct nested dataclass structure
        config = Config(
            simulation=SimulationDefaults(**config_dict.get('simulation', {})),
            ofdm=OFDMDefaults(**config_dict.get('ofdm', {})),
            coding=CodingDefaults(**config_dict.get('coding', {})),
            network=NetworkDefaults(**config_dict.get('network', {})),
            database=DatabaseConfig(**config_dict.get('database', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {}))
        )

        print(f"Configuration loaded successfully from {config_path}")
        return config

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON configuration file: {e}. Using defaults.")
        return Config()
    except TypeError as e:
        print(f"Error in configuration structure: {e}. Using defaults.")
        return Config()
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}. Using defaults.")
        return Config()


def save_config(config: Config, config_path: Optional[str] = None) -> bool:
    """Save configuration to JSON file.

    Serializes the Config object to JSON format and writes to the specified file.
    Creates parent directories if they don't exist. Handles errors gracefully.

    Args:
        config: Config object to save
        config_path: Path where JSON file should be written. If None, uses default
                     'config.json' in the config directory.

    Returns:
        True if save successful, False otherwise.

    Example:
        >>> config = Config()
        >>> config.simulation.snr_db = 15.0
        >>> save_config(config, 'my_config.json')
        True
    """
    # Determine config file path
    if config_path is None:
        config_dir = Path(__file__).parent
        config_path = config_dir / "config.json"
    else:
        config_path = Path(config_path)

    try:
        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert config to dictionary
        config_dict = {
            'simulation': asdict(config.simulation),
            'ofdm': asdict(config.ofdm),
            'coding': asdict(config.coding),
            'network': asdict(config.network),
            'database': asdict(config.database),
            'visualization': asdict(config.visualization)
        }

        # Write to JSON file with pretty formatting
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"Configuration saved successfully to {config_path}")
        return True

    except IOError as e:
        print(f"Error writing configuration file: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error saving configuration: {e}")
        return False


def get_default_config() -> Config:
    """Get a fresh Config object with all default values.

    Convenience function to obtain default configuration without file I/O.

    Returns:
        Config object with all default values.

    Example:
        >>> config = get_default_config()
        >>> config.simulation.modulation_order
        16
    """
    return Config()


# Module-level default config instance for easy import
default_config = get_default_config()


if __name__ == "__main__":
    # Demo usage and testing
    print("=== Orbiter-2 Configuration Module Demo ===\n")

    # Create and display default config
    config = get_default_config()
    print("Default Configuration:")
    print(f"  Sampling Rate: {config.simulation.sampling_rate} Hz")
    print(f"  Modulation Order: {config.simulation.modulation_order}")
    print(f"  SNR: {config.simulation.snr_db} dB")
    print(f"  OFDM Subcarriers: {config.ofdm.num_subcarriers}")
    print(f"  Database Path: {config.database.db_path}\n")

    # Test saving configuration
    test_path = "test_config.json"
    if save_config(config, test_path):
        print(f"\nTest configuration saved to {test_path}")

    # Test loading configuration
    loaded_config = load_config(test_path)
    print(f"\nLoaded configuration from {test_path}")
    print(f"  Loaded SNR: {loaded_config.simulation.snr_db} dB")

    # Cleanup test file
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"\nCleanup: Removed {test_path}")
