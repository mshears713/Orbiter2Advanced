# Orbiter-2: Wireless Communications Simulation Project Overview

## Project Structure

This project follows a modular Python package architecture designed for extensibility, maintainability, and clear separation of concerns.

### Directory Layout

```
wireless_sim/
├── simulations/         # Simulation module implementations
├── visualizations/      # Visualization components
├── models/             # Data models and domain entities
├── db/                 # Database schema and access layer
├── config/             # Configuration management
├── dashboard/          # Streamlit UI application
├── utils/              # Shared utilities and helpers
├── tests/              # Automated test suite
├── examples/           # Standalone usage examples
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
└── notes.txt           # Development notes and version info
```

### Package Descriptions

**simulations/**
Contains all wireless communication algorithm implementations. Each module represents an "alien civilization" teaching a specific wireless concept:
- High-Order Modulation (QAM, PSK)
- OFDM Signal Processing
- Convolutional Coding with Viterbi Decoding
- LDPC Decoding
- Mesh Routing
- Interference Shaping
- Delay-Tolerant Networking
- Adaptive Link Strategies

All simulation modules inherit from a common `SimulationModule` base class, ensuring consistent API across implementations.

**visualizations/**
Houses interactive visualization components using Matplotlib and Plotly. Each visualization corresponds to a simulation module and provides:
- Real-time animated displays
- Interactive controls
- Educational annotations
- Dynamic data updates

All visualizations inherit from a `Visualization` base class with standardized render and update methods.

**models/**
Defines Python dataclasses for domain entities:
- SignalParameters
- ChannelDescription
- SimulationResult
- And other data transfer objects

Uses type hints and validation to ensure data integrity throughout the application.

**db/**
Implements SQLite-based persistence layer:
- Schema definitions (schema.sql)
- Database initialization (init_db.py)
- Access layer with CRUD operations (db_access.py)

Stores simulation metadata, run history, and logs for analysis and debugging.

**config/**
Centralizes all configuration parameters:
- Simulation defaults
- UI settings
- Database paths
- Visualization preferences

Provides load/save functionality for JSON-based configuration files.

**dashboard/**
Contains the Streamlit web application (app.py) that:
- Manages user interactions
- Orchestrates simulation execution
- Embeds visualizations
- Handles session state
- Provides error feedback

**utils/**
Shared utility functions:
- Parameter validation (validation.py)
- Logging setup (logger.py)
- Performance optimization helpers

**tests/**
Comprehensive test suite covering:
- Unit tests for simulations
- Database access tests
- Visualization rendering tests
- Integration tests for dashboard
- End-to-end workflow tests

**examples/**
Standalone scripts demonstrating module usage outside the Streamlit UI.

## Design Principles

### Modularity
Each component has a well-defined interface and can be developed, tested, and extended independently.

### Extensibility
New simulation modules and visualizations can be added by simply implementing the base class interfaces.

### Type Safety
Extensive use of Python type hints and dataclasses ensures type safety and enables better IDE support.

### Error Handling
Comprehensive validation and exception handling at all layers prevents cascading failures.

### Educational Focus
Code is heavily documented with teaching comments explaining wireless communication concepts.

### Performance
Vectorized NumPy operations and optional Numba JIT compilation ensure responsive simulations.

## Data Flow

1. **User Input** → Streamlit UI captures parameters
2. **Validation** → Utils validate inputs before execution
3. **Simulation** → Module runs algorithm asynchronously
4. **Storage** → Database records metadata and logs
5. **Visualization** → Component renders results interactively
6. **Feedback** → UI updates with progress and results

## Development Workflow

1. Create simulation module inheriting from `SimulationModule`
2. Implement corresponding visualization inheriting from `Visualization`
3. Integrate both into dashboard UI
4. Add validation for module-specific parameters
5. Write unit and integration tests
6. Document usage with examples

## Technology Stack

- **Python 3.8+** - Core language
- **Streamlit** - Web UI framework
- **NumPy/SciPy** - Numerical computing
- **Matplotlib/Plotly** - Visualization
- **SQLite** - Embedded database
- **Pandas** - Data manipulation
- **Numba** - Performance optimization (optional)

## Getting Started

See main README.md for setup and usage instructions.

---

**Status**: Project initialized
**Version**: 1.0.0-dev
**Last Updated**: 2025-11-19
