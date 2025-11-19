# Orbiter-2: Deep-Space Wireless Communications Simulation

---

## Overview

Orbiter-2 is a cutting-edge, modular educational software project designed to extend wireless communications learning into the realm of advanced Earth-based technologies through a deep-space themed simulation. It builds upon foundational and intermediate learning arcs by guiding users through sophisticated wireless communication concepts—such as high-order modulation, OFDM internals, convolutional coding, LDPC decoding, mesh routing, interference shaping, delay-tolerant networking, and adaptive link strategies—each uniquely represented by an alien civilization encountered by a CubeSat exploring deep space.

The project employs runnable Python simulation modules tightly integrated with richly animated, interactive visualizations powered by Streamlit, Matplotlib, and Plotly. This immersive approach demystifies complex wireless engineering algorithms by providing hands-on implementations and live data-driven visuals that dynamically illustrate signal behaviors, coding trellises, network topologies, and adaptive radio parameters. Orbiter-2 targets intermediate-level learners and developers with some familiarity in Python and basic wireless concepts, aiming to build deep practical intuition and mental models for modern communications technologies commonly found in WiFi, LTE/5G, LoRa, and mesh networks.

Designed explicitly for autonomous execution by AI agents such as Claude Code, this comprehensive README/PRD guides the AI through a detailed, five-phase, fifty-step construction plan that equips the project with modular codebases, integrated pipelines, interactive dashboards, error handling, extensive testing, documentation, and deployment readiness. Total autonomous execution is anticipated to last 1-3 hours, producing a fully functional, visually rich, and extensible communications learning platform without any human intervention.

---

## Autonomous Execution Guide

**For AI Agents (Claude Code, etc.):**

This README is your complete blueprint for building Orbiter-2 autonomously and sequentially, step-by-step.

**Execution Instructions:**

1. Execute all **five phases** in order, from Phase 1 to Phase 5.
2. Complete all **~50 steps** sequentially, respecting each step's dependencies.
3. Work continuously, **without pausing for user input or approval**.
4. Fully implement all features, visuals, tests, and documentation—**no stubs or placeholders allowed**.
5. Test each component thoroughly before progressing.
6. Persist all work so the final product is sound and ready for delivery.
7. Surface only the complete and working project to the user after all steps are done.

**Execution Strategy:**

- Follow the exact step order (1 → 50), checking dependencies before commencing each step.
- Build robust, production-quality code with comprehensive error handling and comments.
- Produce detailed, self-documenting code with extensive inline teaching commentary.
- Integrate smoothly with database, simulation, visualization, and Streamlit UI layers.
- Ensure all simulation modules and visualizations implement their standardized interfaces.
- Use asynchronous programming or threading where instructed to keep the Streamlit UI responsive.
- Complete all testing scripts (unit, integration, visualization) and logging infrastructure.
- Produce user documentation, examples, and deployment guides for turnkey usage.

---

## Teaching Goals

### Learning Goals

- **Advanced Communications Intuition:** Build hands-on understanding of complex wireless algorithms such as high-order QAM modulation, OFDM signal synthesis, convolutional coding, LDPC message passing, and more.
- **Channel Coding Mastery:** Learn practical encoding and decoding techniques including convolutional codes with Viterbi decoding and LDPC iterative belief propagation.
- **Network Layer Concepts:** Develop intuition for mesh routing protocols, interference shaping methods, delay-tolerant networking, and dynamic link adaptation.
- **Modular Code Design:** Practice architecting extensive, extensible Python projects with clear interfaces enabling pluggable simulation modules.
- **Interactive Visualization Skills:** Gain expertise creating real-time, animated visualizations combining Streamlit with Matplotlib and Plotly for complex system states.

### Technical Goals

- Develop **modular Python simulation modules** encapsulating each advanced wireless technique with clear, type-annotated APIs.
- Construct an **integrated Streamlit dashboard framework** hosting all simulations with dynamic controls and rich interactive visual outputs.
- Implement **dynamic, animated visualization components** (e.g., constellation plots, trellis diagrams, network graphs) that update live during simulations.
- Ensure performance efficiency handling CPU-bound simulations, preserving UI responsiveness with asynchronous patterns and optimized numerical code.

### Priority Notes

- Emphasize modular design patterns and abstraction layers to allow **plug-and-play extensions**.
- Maintain **robust error handling and input parameter validation** to safeguard simulation correctness.
- Extensive **docstrings, inline teaching commentary, and usability-focused design** are required.
- Animations and visuals are **not decorative extras** but core learning tools integral to each simulation.
- The project is deliberately scoped as **educational and intuitive rather than production-grade protocol stacks**.
- The AI agent must produce **fully operational, complete implementations** at every step to ensure autonomous end-to-end success.

---

## Technology Stack

| Layer          | Technology          | Rationale                                                                            |
|----------------|---------------------|-------------------------------------------------------------------------------------|
| Frontend       | **Streamlit**       | Simplifies building interactive dashboards with live data updates and controls. Perfect for educational demos and quick UI iteration. Supports session state and real-time plotting integration. |
| Visualizations | **Matplotlib & Plotly** | Powerful and flexible plotting libraries; Matplotlib for static and animated 2D plots, Plotly for interactive, dynamic visuals supporting hover/zoom features crucial for complex data views. |
| Backend        | **None (in-process)**| All computations performed within the Python app; avoids complexity of backend frameworks. |
| Storage        | **SQLite**          | Embedded database to store simulation metadata and logs; lightweight and easily accessible via Python's builtin `sqlite3`. |
| Utilities      | **pandas, numpy, scipy** | Essential scientific computing libraries for data manipulation, vectorized calculations, and numeric methods fundamental to simulation logic. |
| Performance    | **numba (optional)** | JIT compilation to accelerate CPU-bound numerical simulation code hotspots. |

**Framework Rationale:**

This stack balances accessibility for intermediate Python users with powerful capabilities for scientific computing and interactive visualization. Streamlit provides a low-overhead, user-friendly frontend interface integrated with Python’s full ecosystem, facilitating modular, maintainable codebases. SQLite serves metadata needs simply while advanced plotting libraries enable rich data storytelling. The modular design supports easy extension, while performance tools like numba allow efficient execution of heavy routines.

---

## Architecture Overview

Orbiter-2 is architected as a modular software system comprising distinct layers and packages that interact cohesively:

- **Simulations Package (`simulations/`)**: Contains classes implementing wireless communication algorithms, each adhering to a base interface (`SimulationModule`). Provides `run_simulation(parameters) -> SimulationResult` delivering processed data for visualization.
- **Visualizations Package (`visualizations/`)**: Houses visualization classes conforming to a defined interface (`Visualization`) with methods like `render(simulation_result)` and `update(new_data)` for dynamic display of simulation outputs.
- **Models Package (`models/`)**: Defines Python dataclasses for core domain entities (e.g., signals, channels, simulation results), enabling type-safe data exchange between layers.
- **Database Package (`db/`)**: Implements SQLite schema, access layer (`DBManager`), and ensures persistent metadata and logging.
- **Configuration Package (`config/`)**: Centralizes all simulation and UI configuration parameters for easy tuning and extensibility.
- **Dashboard Package (`dashboard/`)**: Consists of the Streamlit app components, managing user input, simulation execution, visualization embedding, and session state handling.
- **Utilities Package (`utils/`)**: Provides shared helper modules such as parameter validation, logging setup, and performance optimizations.

---

### Data Flow Overview

1. **User Interaction via Streamlit UI**: Users select simulations and input parameters via sidebar controls.
2. **Input Validation**: Parameters are checked using validation utilities before simulation execution.
3. **Simulation Execution**: The selected simulation module runs asynchronously, producing simulation results.
4. **Data Storage**: Metadata about simulation runs and logs are stored in SQLite.
5. **Visualization Rendering**: Corresponding visualization modules render and animate simulation outputs in real-time.
6. **Session Management**: Streamlit session state maintains parameters, selections, and simulation status across reruns.
7. **Continuous Feedback**: UI updates dynamically with progress spinners, error messages, and animated visual outputs.

---

### Directory Structure

```
wireless_sim/
├── simulations/
│   ├── base.py
│   ├── high_order_modulation.py
│   ├── ofdm.py
│   ├── convolutional_coding.py
│   ├── ldpc.py
│   ├── mesh_routing.py
│   ├── interference.py
│   ├── dtn.py
│   ├── adaptive_link.py
│   └── exceptions.py
├── visualizations/
│   ├── base.py
│   ├── high_order_constellation.py
│   ├── ofdm_constellation.py
│   ├── convolutional_trellis.py
│   ├── ldpc_iteration.py
│   ├── mesh_network.py
│   ├── interference_heatmap.py
│   ├── dtn_metrics.py
│   ├── adaptive_timeseries.py
│   └── placeholder_viz.py
├── models/
│   └── datamodels.py
├── db/
│   ├── schema.sql
│   ├── init_db.py
│   └── db_access.py
├── config/
│   └── config.py
├── dashboard/
│   └── app.py
├── utils/
│   ├── validation.py
│   └── logger.py
├── examples/
│   └── run_high_order_modulation.py
├── tests/
│   ├── test_db_access.py
│   ├── test_simulation_api.py
│   ├── test_visualizations.py
│   └── test_dashboard_integration.py
├── main.py
├── README.md
├── requirements.txt
└── notes.txt
```

---

### ASCII Diagram of Component Interaction

```
+--------------------+       +-----------------------+        +--------------------+
|                    |       |                       |        |                    |
|  User (Streamlit)   +------>+  Dashboard / UI Layer  +------->+  Simulation Modules | 
|                    |       | (app.py + controls)    |        | (simulations/*)     |
+--------------------+       +--------+--------------+        +---------+----------+
                                             |                             |
                                             | Parameters / requests       |
                                             v                             v
                                  +-------------------+       +-------------------+
                                  | Visualization     |<------+ Simulation Results |
                                  | Modules           |       |                   |
                                  | (visualizations/*)|       +-------------------+
                                  +-------------------+ 
                                             |
                                             v
                                  +-------------------+
                                  | SQLite Database &  |
                                  | Logging (db/*)     |
                                  +-------------------+
```

---

## Implementation Plan

**Execute phases sequentially, fully completing one phase before moving on. Each step is a fully operational, tested unit with documented code and no placeholders.**

---

### Phase 1: Foundations & Environment Setup (Steps 1-10)

**Overview:**

Establish project groundwork: directory structure, environment, configuration, data models, database setup, basic simulation and visualization interfaces, and the initial Streamlit dashboard skeleton with input validation utilities.

**Completion Criteria:**

- Modular project structure created with all packages and init files.
- Virtual environment and dependencies installed.
- Config module centralizing parameters.
- Core dataclasses representing domain models.
- SQLite schema and DB access layer operational.
- Abstract simulation and visualization base classes defined.
- Initial Streamlit dashboard framework with navigation placeholders.
- Robust parameter validation utilities implemented.
- Preliminary tests covering DB and validation modules included.

---

#### Step 1: Initialize Python Project Structure with Modular Package Layout

**What to Build:**

- Create root directory `wireless_sim/`
- Create subdirectories: `simulations/`, `visualizations/`, `models/`, `db/`, `config/`, `dashboard/`, `utils/`, `tests/`, `examples/`
- Add `__init__.py` to each to define packages.
- Add a placeholder `README.md` at root.
- Create empty `main.py` for launching dashboard.
- Write a `project_overview.md` summarizing directory layout and project intent.

**Implementation Details:**

- Use standard Python package conventions.
- Each package must have clear separation of concerns.
- Provide docstrings or markdown explaining purpose of directories.
- Add comments outlining where simulation modules, visualizations, and tests belong.
- Create ignore files (e.g., `.gitignore`) for environment dirs if needed.

**Dependencies:** None

**Acceptance Criteria:**

- Project directory and all packages exist with `__init__.py`.
- README placeholder created.
- `main.py` file exists.

---

#### Step 2: Set Up Virtual Environment and Install Dependencies

**What to Build:**

- Create Python virtual environment under `./venv`.
- Add `requirements.txt` listing pinned versions:
  - numpy
  - scipy
  - matplotlib
  - plotly
  - streamlit
  - pandas
  - numba (optional)
- Automate environment setup steps in a `setup.py` or instruction script.
- Add `notes.txt` recording package versions.
- Include version compatibility check for Python 3.8+.

**Implementation Details:**

- Use `python3 -m venv venv` and activation instructions for Unix/Windows.
- Use `pip install -r requirements.txt`.
- Implement basic error handling in setup script for failed installs.
- Document all commands in `notes.txt`.
- Validate Python version early with exception messaging.

**Dependencies:** Step 1

**Acceptance Criteria:**

- Virtual environment folder created.
- All dependencies installed successfully without errors.
- `notes.txt` contains version info and setup steps.

---

#### Step 3: Design and Code Basic Configuration Module

**What to Build:**

- `config/config.py` defining a `Config` class or constants.
- Provide parameters such as sampling_rate, modulation_order defaults, db path, visualization settings.
- Functions: `load_config()`, `save_config()` supporting JSON/YAML files.
- Robust error handling to fall back on defaults if config file missing or malformed.
- Extensive docstrings explaining each config parameter and usage.

**Implementation Details:**

- Use Python’s `json` or `yaml` for config persistence.
- Clearly separate simulation parameters from UI settings.
- Include usage examples in docstrings.

**Dependencies:** Step 1

**Acceptance Criteria:**

- Config module loads default settings successfully.
- Reading/writing config files works correctly.
- Code includes clear documentation.

---

#### Step 4: Define Core Data Models Using Python Dataclasses

**What to Build:**

- `models/datamodels.py` with dataclasses:
  - `SignalParameters`: modulation_order (int), bandwidth (float), snr (float)
  - `ChannelDescription`: type (str), fading (bool), usage_notes (str)
  - `SimulationResult`: timestamp (datetime), parameters (SignalParameters), data (np.ndarray)
- Add type checking, default values, and validation in `__post_init__`.
- Serialization methods to/from JSON if needed.
- Comprehensive docstrings and inline comments.

**Implementation Details:**

- Use `dataclasses` module with type hints.
- Validate modulation_order is a power of two.
- Design with future extensibility in mind.

**Dependencies:** Step 1

**Acceptance Criteria:**

- Dataclasses declared correctly.
- Validation triggers for invalid input.
- Serialization methods functional.

---

#### Step 5: Set Up Basic SQLite Database Schema for Simulation Metadata

**What to Build:**

- `db/schema.sql` defining:
  - `simulation_runs` table: id, start_time, end_time, parameters (JSON), success flag.
  - `logs` table: id, sim_run_id (FK), log_level, message, timestamp.
- Python script `db/init_db.py` with `init_db(db_path)` to create DB and tables if missing.
- Proper error handling for DB connection and schema errors.
- Documentation comments explaining schema design.

**Implementation Details:**

- Use SQLite’s `sqlite3` module with context managers.
- Store parameters JSON as TEXT.
- Foreign key constraints used.

**Dependencies:** Step 1

**Acceptance Criteria:**

- Database file created with correct tables.
- `init_db` callable without errors on existing/new DB.
- Schema matches specification.

---

#### Step 6: Implement Database Access Layer with CRUD Functions

**What to Build:**

- `db/db_access.py` defining `DBManager` class with methods:
  - `__init__(db_path)`
  - `insert_simulation_run(start_time, parameters) → sim_id`
  - `update_simulation_run(sim_id, end_time, success)`
  - `insert_log(sim_id, level, message)`
  - `fetch_simulation_run(sim_id) → dict`
- Use parameterized SQL queries and exception handling.
- Serialize parameters dict to JSON on insert.
- Unit tests for CRUD operations (`tests/test_db_access.py`).

**Implementation Details:**

- Wrap SQLite operations with try-except for operational errors.
- Use connection pooling/singleton if beneficial.
- Document methods with complete type hints and docstrings.

**Dependencies:** Step 5

**Acceptance Criteria:**

- DBManager methods work as expected.
- Unit tests pass for all CRUD functionalities.
- Exceptions handled gracefully.

---

#### Step 7: Create a Placeholder Simulation Module Interface

**What to Build:**

- `simulations/base.py` defining abstract `SimulationModule` base class with:
  - `run_simulation(parameters: dict) -> SimulationResult`
  - `get_name() -> str`
- Use `abc` module and proper typing.
- Define a custom `SimulationError` exception for common error handling.
- Add a minimal example placeholder module (`simulations/placeholder_sim.py`) inheriting the base.
- Document expected inputs, outputs, errors, and interface requirements clearly.

**Implementation Details:**

- Use Python’s `abc.ABC` and `abc.abstractmethod`.
- Include detailed docstrings and comments explaining subclass requirements.

**Dependencies:** Step 4

**Acceptance Criteria:**

- Base class enforces interface.
- Example subclass compiles and runs returning a dummy SimulationResult.
- Clear documentation included.

---

#### Step 8: Set Up Base Visualization Module Interface

**What to Build:**

- `visualizations/base.py` defining abstract base class `Visualization` with:
  - `render(simulation_result: SimulationResult) -> None`
  - `update(new_data: dict) -> None`
- Design to integrate with Matplotlib or Plotly backends.
- Include base utilities for layout initialization.
- Provide placeholder visualization implementation (`visualizations/placeholder_viz.py`).
- Comprehensive docstrings describing rendering lifecycle, update logic, and exceptions.

**Implementation Details:**

- Abstract methods to enforce subclass implementation.
- Explain usage in code comments.

**Dependencies:** Step 4

**Acceptance Criteria:**

- Visualization base class created with documented methods.
- Placeholder visualization functions without error.
- Documentation suffices for future implementers.

---

#### Step 9: Initialize Basic Streamlit Dashboard Framework

**What to Build:**

- `dashboard/app.py` implementing Streamlit application entry point.
- Sidebar with selection controls for simulation modules (initially empty/placeholders).
- Main page with headers, input parameter placeholders, simulation start/stop buttons, and visualization container placeholders (using `st.empty()`).
- Implement session state to remember UI selections.
- Architect dynamic module import for future integration.
- Use `st.spinner()` loading animations and error handling panels.
- Comprehensive inline comments for UI logic, state management, and extension hooks.
- Ensure app runs without errors and displays basic UI skeleton.

**Implementation Details:**

- Utilize Streamlit widgets: `st.sidebar.radio()`, `st.button()`, `st.container()`.
- Initialize session state variables on first load.
- Prepare for module integration but build minimal scaffolding now.

**Dependencies:** Step 3

**Acceptance Criteria:**

- Streamlit app launches successfully without errors.
- Sidebar and main area display placeholders.
- Session state persists user selections.

---

#### Step 10: Implement Utility Functions for Simulation Parameter Validation

**What to Build:**

- `utils/validation.py` with reusable functions:
  - `validate_modulation_order(order: int) -> bool`
  - `validate_snr(snr_db: float) -> bool`
  - `validate_parameters(params: dict) -> Tuple[bool, List[str]]`
- Bounds checking (e.g., modulation order is a power of two, SNR in realistic range).
- Return clear error messages or raise exceptions.
- Include doctests demonstrating validation success and failure.
- Integrate validation functions into Streamlit dashboard form inputs for live feedback.

**Implementation Details:**

- Use standard Python exceptions or custom validation error classes.
- Return tuples of success flag and error list.
- Document rationale and usage patterns in docstrings.

**Dependencies:** Step 4

**Acceptance Criteria:**

- Validation functions return expected results for valid and invalid inputs.
- Doctests pass.
- Functions integrate into UI validation seamlessly.

---

### Phase 2: Core Simulation Modules Development (Steps 11-20)

**Overview:**

Develop the first three alien civilization modules representing advanced wireless techniques: High-Order Modulation, OFDM, and Convolutional Coding. Each includes simulation logic, dynamic visualizations, and dashboard integration ensuring responsive user experience and error handling.

**Completion Criteria:**

- Fully functional high-order modulation simulation with noise and constellation animation.
- OFDM signal processing modeled with correct subcarrier handling and constellation visualization.
- Rate 1/2 convolutional coding and Viterbi decoding simulation with animated trellis visualization.
- Dashboard integration including parameter controls, visualization embedding, asynchronous simulation runs, and UI error feedback.

---

#### Step 11: Implement 'Alien Civilization 1: High-Order Modulation Simulation

**What to Build:**

- `simulations/high_order_modulation.py` with `HighOrderModulationSimulation` implementing `SimulationModule`.
- `run_simulation(parameters: dict) → SimulationResult` does:
  - Generate random bit stream.
  - Map bits to modulation symbols (QAM/PSK) based on modulation order (16-QAM, 64-QAM, etc.).
  - Add AWGN noise with specified SNR.
  - Return constellation points as numpy arrays.
- Use numpy vectorized methods.
- Handle errors for unsupported modulation orders or negative SNRs with `SimulationError`.
- Add debug logging.

**Implementation Details:**

- Implement mathematical mapping functions per modulation scheme.
- Validate input parameters via utilities.
- Add comments explaining signal processing steps.

**Dependencies:** Steps 7, 10

**Acceptance Criteria:**

- Module runs without error with valid parameters.
- Returns correct constellation data in SimulationResult.
- Errors raised for invalid inputs.

---

#### Step 12: Create Animated Constellation Plot Visualization for Modulation

**What to Build:**

- `visualizations/high_order_constellation.py` with `HighOrderConstellationPlot` subclassing `Visualization`.
- Use Matplotlib to plot scatter diagrams of transmitted and noisy received symbols.
- Implement `render()` to initialize plot layout, titles, and axes.
- Implement `update(new_data: dict)` to animate adding received symbols gradually.
- Include legends, axis labels, and titles.
- Handle incorrect data shapes with exceptions.
- Document expected input format.

**Implementation Details:**

- Use `plt.pause()` or integrate updates compatible with Streamlit.
- Map modulation order to standard constellation layout inside visualization.

**Dependencies:** Steps 8, 11

**Acceptance Criteria:**

- Visualization displays transmitted and received constellation points.
- Animations appear smooth and update dynamically.
- Exceptions raised on malformed data.

---

#### Step 13: Integrate High-Order Modulation Module and Visualization into Streamlit App

**What to Build:**

- Modify `dashboard/app.py`:
  - Add sidebar controls: modulation order selector, SNR slider, symbol count input.
  - Use Streamlit forms and input validation using `utils.validation`.
  - On form submission, launch `HighOrderModulationSimulation.run_simulation` asynchronously.
  - Render `HighOrderConstellationPlot` in main area with live updates.
  - Handle exceptions and show `st.error()` messages.
  - Preserve input params with session state.
  - Explain code flow in inline comments.

**Implementation Details:**

- Use `concurrent.futures.ThreadPoolExecutor` or Streamlit's cache and async features.
- Provide loading spinners with `st.spinner()`.
- Ensure cancellation or resetting capability.

**Dependencies:** Steps 9, 11, 12

**Acceptance Criteria:**

- Users can configure and run high-order modulation simulations without UI blocking.
- Visualizations display interactively.
- Error conditions handled gracefully.

---

#### Step 14: Implement 'Alien Civilization 2: OFDM Signal Processing Module

**What to Build:**

- `simulations/ofdm.py` with `OFDMSimulation`.
- Simulate OFDM symbols:
  - Generate bits per subcarrier using modulation order.
  - Apply IFFT for time-domain OFDM symbols.
  - Add cyclic prefix.
  - Add AWGN noise.
- Parameters: number of subcarriers, cyclic prefix length, modulation order per subcarrier.
- Placeholder for channel effects.
- Validate inputs with exceptions.
- Document steps and theory.

**Dependencies:** Step 7

**Acceptance Criteria:**

- OFDM simulation produces time-domain OFDM signals.
- Modular and extensible code.
- Throws errors for invalid parameters.

---

#### Step 15: Create OFDM Subcarrier Constellation Visualization

**What to Build:**

- `visualizations/ofdm_constellation.py` implementing `OFDMSuperImposedConstellation`.
- Use Plotly scatter plots showing subcarrier constellations distinctively.
- Animate temporal evolution of symbols per subcarrier.
- Tooltips display subcarrier index and symbol.
- Exception handling for missing data.
- Inline documentation for update logic.

**Dependencies:** Steps 8, 14

**Acceptance Criteria:**

- Visual shows distinguishable points for each subcarrier.
- Animation runs smoothly.
- Errors handled gracefully.

---

#### Step 16: Integrate OFDM Simulation and Visualization in Dashboard

**What to Build:**

- Extend `dashboard/app.py`:
  - Panel for OFDM inputs: number of subcarriers, cyclic prefix length, modulation order.
  - Validate and preserve inputs.
  - Run simulation asynchronously.
  - Pass results to `OFDMSuperImposedConstellation.render`.
  - UI supports switching and state persistence.
  - Detailed comments explaining UI-simulation coupling.

**Dependencies:** Steps 9, 14, 15

**Acceptance Criteria:**

- OFDM module runs interactively inside dashboard.
- Visualization embedded and animated live.
- Input errors caught and user informed.

---

#### Step 17: Implement 'Alien Civilization 3: Convolutional Coding Module

**What to Build:**

- `simulations/convolutional_coding.py` with `ConvolutionalCodingSimulation`.
- Encode binary inputs via rate 1/2 convolutional encoder (constraint length 7, polynomials 133o, 171o).
- Decode noisy channel outputs using soft-decision Viterbi.
- `run_simulation` simulates encoding, AWGN noise addition, and decoding.
- Validate parameters strictly.
- Detailed comments on trellis and decoding logic.
- Raise `SimulationError` on invalid inputs.

**Dependencies:** Step 7

**Acceptance Criteria:**

- Encoding/decoding functions produce correct decoded output.
- Handles noise and produces metrics.
- Error handling present.

---

#### Step 18: Develop Trellis Diagram Dynamic Visualization for Convolutional Coding

**What to Build:**

- `visualizations/convolutional_trellis.py` with `ConvolutionalTrellisPlot`.
- Matplotlib animated plot of trellis showing states, path metrics, and survivor paths.
- `render` sets up diagram; `update` advances animation frames.
- Embed in Streamlit with `st.pyplot`.
- Input validation and exception handling.

**Dependencies:** Steps 8, 17

**Acceptance Criteria:**

- Trellis diagram animates decoding progression.
- Shows state updates and path metrics.
- Works without exceptions on valid data.

---

#### Step 19: Add Convolutional Coding Simulation and Trellis Visualization to Dashboard

**What to Build:**

- Add convolutional coding UI panel to `dashboard/app.py`.
- Controls for generator polynomials, bitstream length, noise level.
- Async simulation invocation.
- Stream results to trellis plot.
- Maintain session state, handle errors, add spinners.
- Document integration with ample comments.

**Dependencies:** Steps 9, 17, 18

**Acceptance Criteria:**

- Users can run convolutional coding simulations interactively.
- Trellis visualization updates live.
- Input validation present.

---

#### Step 20: Design Simulation Module API for Easy Plug-and-Play Extensions

**What to Build:**

- Extend `simulations/base.py`:
  - Formalize mandatory methods:
    - `get_name() -> str`
    - `run_simulation(params: dict) -> SimulationResult`
    - `get_default_parameters() -> dict`
    - `get_parameter_schema() -> dict`
  - Use `pydantic` or `jsonschema`-style parameter schemas for validation.
- Add detailed docstrings and inline comments explaining API usage.
- Write automated tests (`tests/test_simulation_api.py`) verifying modules conform.
- Enforce consistent interface for future extensibility.

**Dependencies:** Steps 7, 8

**Acceptance Criteria:**

- Base API documented and enforced.
- Tests confirm conformity for existing modules.
- Future modules comply seamlessly.

---

### Phase 3: Advanced Modules & Simulation Integration (Steps 21-35)

**Overview:**

Implement the remaining alien civilization modules encompassing LDPC decoding, mesh routing, interference shaping, delay-tolerant networking, and adaptive link strategies. Develop their sophisticated animations and integrate visually/operationally into the dashboard framework.

**Completion Criteria:**

- All advanced simulation modules implemented with full parameter handling and validation.
- Corresponding animated visualizations created, efficient and smooth.
- Dashboard extended to support all new modules with input controls, asynchronous simulations, and visualization embedding.
- Cross-module UI consistency maintained.

(For brevity, only titles and key points. Details follow same conventions as Phases 1-2, steps per project plan.)

- Step 21: LDPC Decoding module simulating iterative belief propagation.
- Step 22: Visualization of LDPC decoding iteration heatmaps.
- Step 23: LDPC module dashboard integration.
- Step 24: Mesh Routing simulation implementing network graph modeling and routing algorithms.
- Step 25: Mesh routing network graph visualization with packet animation.
- Step 26: Mesh routing dashboard UI embedding.
- Step 27: Interference shaping simulation generating interference spatial patterns.
- Step 28: Heatmap visualization of interference patterns.
- Step 29: Interference shaping UI and visualization dashboard integration.
- Step 30: Delay-Tolerant Networking simulation modeling asynchronous, store-carry-forward routing.
- Step 31: DTN delivery delay and success rate metrics visualization.
- Step 32: DTN module seamless dashboard integration.
- Step 33: Adaptive Link Strategy simulation with dynamic modulation/power adjustment.
- Step 34: Adaptive parameter time-series visualization.
- Step 35: Adaptive link strategy controls and visualization in dashboard.

---

### Phase 4: Polish, Testing & Performance Optimization (Steps 36-45)

**Overview:**

Improve all modules for robustness, performance, UI fluidity; add comprehensive error handling, input sanitization, logging, automated testing (unit, integration, visualization), session state management, graceful shutdowns, and end-to-end integration testing.

**Completion Criteria:**

- Simulation and visualization modules error handled gracefully with precise exceptions.
- Input validation embedded in dashboard prevents invalid runs.
- Optimized numerical code (vectorized, numba-accelerated) deployed.
- Smooth and responsive animations with thoughtful redraw optimizations.
- Complete suite of automated tests operational.
- Comprehensive logging infrastructure records simulation and user activities.
- Streamlit session state fully utilized for persistent user settings.
- Clean resource shutdown implemented.
- Full integration tests verify dashboard functionality and stability.

---

### Phase 5: Documentation, Examples & Deployment Preparation (Steps 46-50)

**Overview:**

Produce polished documentation including extensive README, per-module user guides, enriched inline code comments, example scripts illustrating usage outside Streamlit, and detailed instructions for deployment and dependency management.

**Completion Criteria:**

- Detailed top-level README.md describing project goals, setup, usage, and modules.
- User guides for each simulation module with thorough explanations and troubleshooting.
- Complete inline docstrings and code comments throughout codebase.
- Example scripts demonstrating modular usage runnable independently.
- Deployment instructions with frozen requirements.txt, optional Dockerfile, and setup notes for robust installation and launch.

---

## Implementation Strategy for AI Agents

- Begin with **Phase 1, Step 1**, completing each step fully before proceeding.
- Always check dependencies before implementation.
- Build modular, fully functional code—no stubs.
- Test components incrementally using provided unit and integration tests.
- Add exhaustive code comments and docstrings for maintainability and teaching.
- Use asynchronous or threaded models in dashboard for simulation runs.
- Implement robust error handling end-to-end.
- Maintain Streamlit session state to preserve user inputs and visual states.
- Fully integrate simulation modules and visualizations into the dashboard UI as you progress.
- Final deliverable is a polished, tested, documented codebase and interactive app requiring no further input.

---

## Setup Instructions

- **Python Version:** 3.8 or higher (preferably 3.10+)
- **Virtual Environment Setup:**

  ```bash
  python3 -m venv venv
  source venv/bin/activate  # For Unix/macOS
  venv\Scripts\activate     # For Windows
  pip install -r requirements.txt
  ```

- **Dependencies:** All pinned in `requirements.txt` including numpy, scipy, matplotlib, plotly, streamlit, pandas, and numba (optional).
- **Configuration:** Use `config/config.py` for default parameters; override with JSON config files as needed.
- **Database Initialization:** Execute `db/init_db.py` to create SQLite DB with schema before running simulations.
- **Launching App:**

  ```bash
  streamlit run dashboard/app.py
  ```

- **Examples:** Found in `examples/` directory with instructions on running standalone simulation scripts.
- **Environment Variables:** None required for default setup.
- **Project Directory:** As defined in architecture overview; ensure all package folders and files exist.

---

## Testing Strategy

- **Unit Tests:** Located in `tests/` covering DB access, simulation logic, validation utilities, and visualization rendering.
- **Integration Tests:** Ensure end-to-end workflows, especially dashboard control-simulation-visualization integration.
- **Visualization Tests:** Verify rendering updates and response to changing data inputs.
- **Manual Testing:** Use Streamlit interactive UI to validate user flows.
- **Edge Cases:** Validate handling of invalid inputs, empty data sets, simulation failures.
- **Test Execution:**

  ```bash
  pytest tests/
  ```

- Tests should pass without failures or errors.

---

## Success Metrics

- Complete implementation of all **50 steps**.
- All **five phases** deliver working functionality as described.
- Project runs **without runtime errors**.
- Streamlit UI interacts smoothly, respecting input validations.
- Visualizations animate fluently and correctly represent data.
- Codebase has **comprehensive documentation and comments**.
- Automated tests cover core logic and UI components.
- User documentation and example scripts are complete and clear.
- Deployment instructions allow reproducible installations.
- No `TODO` or placeholder code remains.

---

## Project Completion Checklist

- [ ] All 50 steps completed sequentially.
- [ ] Project directories and modular packages fully implemented.
- [ ] Virtual environment and dependencies installed successfully.
- [ ] Configuration module with load/save functionality done.
- [ ] Core dataclasses defined with validation.
- [ ] SQLite database schema and ORM layer functional.
- [ ] Abstract simulation and visualization base classes created.
- [ ] Streamlit dashboard skeleton with session state set up.
- [ ] Input validation utilities written and integrated.
- [ ] High-order modulation simulation module and constellation visualization implemented and integrated.
- [ ] OFDM simulation and visualization implemented and integrated.
- [ ] Convolutional coding simulation and trellis visualization implemented and integrated.
- [ ] Simulation Module API formalized and enforced.
- [ ] Remaining advanced modules (LDPC, mesh routing, interference shaping, DTN, adaptive links) implemented with visualizations and dashboard support.
- [ ] Error handling, input sanitization, and logging implemented project-wide.
- [ ] Performance optimizations via profiling and numba applied.
- [ ] Visualization update logic refined for smooth animations.
- [ ] Complete suite of automated tests developed and passing.
- [ ] Streamlit session state effectively used for UI persistence.
- [ ] Graceful shutdown and resource cleanups implemented.
- [ ] End-to-end integration tests of dashboard passed.
- [ ] Comprehensive README and per-module user guides written.
- [ ] Codebase fully commented and docstrings compliant with standards.
- [ ] Example scripts for each module created.
- [ ] Deployment instructions and frozen requirements documented.
- [ ] Final program runs and delivers all specified features successfully.

---

# End of README - Ready for Autonomous Execution by AI Agent
