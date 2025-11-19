# Claude Development Guide for Orbiter-2

## Project Overview
Orbiter-2 is an educational deep-space wireless communications simulation platform built with Python, Streamlit, and scientific computing libraries. It teaches advanced wireless concepts through interactive visualizations representing alien civilizations.

## Architecture Summary

### Core Packages
- `simulations/` - Simulation modules implementing wireless algorithms
- `visualizations/` - Interactive visualization components
- `models/` - Data models using dataclasses
- `db/` - SQLite database layer
- `config/` - Configuration management
- `dashboard/` - Streamlit UI application
- `utils/` - Validation and logging utilities
- `tests/` - Automated test suite
- `examples/` - Standalone usage examples

### Key Technologies
- **Frontend**: Streamlit
- **Visualizations**: Matplotlib, Plotly
- **Storage**: SQLite
- **Scientific**: NumPy, SciPy, Pandas
- **Performance**: Numba (optional)

## Implementation Approach

### Execution Strategy (50 Steps, 5 Phases)

**Phase 1: Foundations (Steps 1-10)**
- Project structure setup
- Environment configuration
- Database schema
- Base interfaces for simulations/visualizations
- Basic Streamlit dashboard skeleton
- Input validation utilities

**Phase 2: Core Modules (Steps 11-20)**
- High-Order Modulation simulation + constellation viz
- OFDM signal processing + subcarrier viz
- Convolutional Coding + trellis viz
- Dashboard integration for each module

**Phase 3: Advanced Modules (Steps 21-35)**
- LDPC decoding + iteration viz
- Mesh routing + network graph viz
- Interference shaping + heatmap viz
- DTN (Delay-Tolerant Networking) + metrics viz
- Adaptive link strategy + timeseries viz

**Phase 4: Polish & Testing (Steps 36-45)**
- Error handling and input sanitization
- Performance optimization (vectorization, numba)
- Comprehensive test suite
- Logging infrastructure
- Session state management

**Phase 5: Documentation (Steps 46-50)**
- Complete README
- Module user guides
- Example scripts
- Deployment instructions

## Key Design Principles

### Module Interface Pattern
All simulation modules inherit from `SimulationModule` base class:
- `run_simulation(parameters: dict) -> SimulationResult`
- `get_name() -> str`
- `get_default_parameters() -> dict`
- `get_parameter_schema() -> dict`

All visualizations inherit from `Visualization` base class:
- `render(simulation_result: SimulationResult) -> None`
- `update(new_data: dict) -> None`

### Data Flow
1. User inputs parameters via Streamlit UI
2. Parameters validated using `utils.validation`
3. Simulation runs asynchronously (non-blocking UI)
4. Results stored in SQLite database
5. Visualizations render with live updates
6. Session state preserves user selections

### Critical Requirements
- **No placeholders or stubs** - fully functional code only
- **Comprehensive error handling** - validate all inputs
- **Extensive documentation** - docstrings and inline comments
- **Async execution** - keep UI responsive during simulations
- **Type annotations** - use type hints throughout
- **Testing** - unit, integration, and visualization tests

## Development Workflow

### For Each Module Implementation:
1. Create simulation class inheriting from `SimulationModule`
2. Implement core algorithm with numpy vectorization
3. Add parameter validation and error handling
4. Create visualization class inheriting from `Visualization`
5. Implement rendering and animation logic
6. Integrate into Streamlit dashboard with controls
7. Add unit tests for simulation logic
8. Test UI integration end-to-end

### Code Quality Standards
- Use type hints for all function signatures
- Add comprehensive docstrings (Google style)
- Include inline teaching comments explaining algorithms
- Validate inputs early with clear error messages
- Use context managers for resource management
- Follow PEP 8 style guidelines

## Common Patterns

### Parameter Validation
```python
from utils.validation import validate_parameters

def run_simulation(self, params: dict) -> SimulationResult:
    is_valid, errors = validate_parameters(params)
    if not is_valid:
        raise SimulationError(f"Invalid parameters: {errors}")
    # ... proceed with simulation
```

### Database Operations
```python
from db.db_access import DBManager

db = DBManager(config.DB_PATH)
sim_id = db.insert_simulation_run(start_time, parameters)
# ... run simulation
db.update_simulation_run(sim_id, end_time, success=True)
```

### Streamlit Integration
```python
# Use session state for persistence
if 'selected_module' not in st.session_state:
    st.session_state.selected_module = None

# Async simulation execution
with st.spinner("Running simulation..."):
    result = module.run_simulation(params)

# Error handling
try:
    visualization.render(result)
except Exception as e:
    st.error(f"Visualization error: {e}")
```

## Progress Tracking

Use TodoWrite tool to track implementation progress through all 50 steps. Mark steps as:
- `pending` - Not started
- `in_progress` - Currently working on
- `completed` - Fully implemented and tested

## Testing Strategy

- **Unit tests**: Test individual simulation algorithms
- **Integration tests**: Test module + visualization + dashboard
- **Visualization tests**: Verify rendering without errors
- **Manual tests**: Interactive UI testing via Streamlit

Run tests with: `pytest tests/`

## Deployment

```bash
# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Initialize database
python db/init_db.py

# Launch application
streamlit run dashboard/app.py
```

## Success Criteria
- All 50 steps completed sequentially
- No runtime errors
- All tests passing
- Smooth, responsive UI
- Complete documentation
- No TODO comments or placeholders

---

**Development Status**: Ready to begin Phase 1, Step 1
**Current Branch**: claude/setup-project-docs-012gcTXGfTM5yPf8q8AL9ANo
**Autonomous Execution**: Enabled - proceed through all steps without pausing
