"""
Orbiter-2 Streamlit Dashboard Application

Main entry point for the interactive web-based simulation dashboard.
Provides UI for selecting simulations, configuring parameters, running
simulations, and viewing animated visualizations.

Author: Claude (Orbiter-2 AI Implementation)
Date: 2025-11-19
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import load_config
from db.db_access import DBManager
from datetime import datetime


# Page configuration
st.set_page_config(
    page_title="Orbiter-2: Wireless Communications Simulation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'selected_simulation' not in st.session_state:
        st.session_state.selected_simulation = None
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    if 'db' not in st.session_state:
        st.session_state.db = DBManager()


def render_header():
    """Render application header."""
    st.title("🛰️ Orbiter-2: Deep-Space Wireless Communications Simulation")
    st.markdown("""
    **Educational wireless communications platform exploring advanced techniques
    through interactive simulations and visualizations.**

    Each module represents an alien civilization's unique wireless technology encountered
    during a CubeSat's deep-space journey.
    """)
    st.divider()


def render_sidebar():
    """Render sidebar with navigation and controls."""
    with st.sidebar:
        st.header("🎛️ Control Panel")

        # Available simulation modules (placeholder - will be populated dynamically later)
        available_sims = [
            "Welcome",
            "High-Order Modulation (Civilization 1)",
            "OFDM Signal Processing (Civilization 2)",
            "Convolutional Coding (Civilization 3)",
            "LDPC Decoding (Civilization 4)",
            "Mesh Routing (Civilization 5)",
            "Interference Shaping (Civilization 6)",
            "Delay-Tolerant Networking (Civilization 7)",
            "Adaptive Link Strategy (Civilization 8)"
        ]

        st.subheader("Select Simulation")
        selected = st.selectbox(
            "Choose a simulation module:",
            available_sims,
            index=0
        )

        st.session_state.selected_simulation = selected

        st.divider()

        # Database statistics
        st.subheader("📊 Statistics")
        try:
            stats = st.session_state.db.get_statistics()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Runs", stats.get('total_runs', 0))
            with col2:
                st.metric("Successful", stats.get('successful_runs', 0))

            if stats.get('avg_execution_time_ms', 0) > 0:
                st.metric(
                    "Avg Time",
                    f"{stats['avg_execution_time_ms']:.1f} ms"
                )
        except Exception as e:
            st.warning(f"Could not load statistics: {e}")

        st.divider()

        # Configuration section
        with st.expander("⚙️ Configuration"):
            st.info("Configuration settings loaded from config.json")
            config = st.session_state.config
            st.write(f"**Sampling Rate:** {config.simulation.sampling_rate / 1e6:.1f} MHz")
            st.write(f"**Default SNR:** {config.simulation.snr_db} dB")
            st.write(f"**Database:** {config.database.db_path}")

        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Orbiter-2** is an educational platform for learning advanced
            wireless communications concepts through interactive simulations.

            **Features:**
            - Real-time animated visualizations
            - Comprehensive parameter controls
            - Performance metrics tracking
            - Educational commentary

            **Version:** 1.0.0-dev
            """)


def render_welcome_page():
    """Render welcome/landing page."""
    st.header("Welcome to Orbiter-2")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🚀 Mission Overview

        You are piloting a CubeSat on a deep-space exploration mission. During your journey,
        you encounter eight alien civilizations, each using unique wireless communication
        technologies. Your mission: learn and master each civilization's techniques.

        ### 📡 Available Simulations

        Each simulation module teaches a specific wireless communications concept:

        1. **High-Order Modulation** - Learn QAM and PSK modulation schemes with
           constellation diagrams showing symbol mapping and noise effects.

        2. **OFDM Signal Processing** - Explore orthogonal frequency division multiplexing
           with subcarrier visualization and cyclic prefix demonstration.

        3. **Convolutional Coding** - Master forward error correction with animated
           trellis diagrams showing Viterbi decoding in action.

        4. **LDPC Decoding** - Understand iterative belief propagation with heatmaps
           showing convergence behavior.

        5. **Mesh Routing** - Simulate multi-hop wireless networks with dynamic
           routing and topology visualization.

        6. **Interference Shaping** - Control spatial interference patterns with
           beamforming and nulling techniques.

        7. **Delay-Tolerant Networking** - Explore store-and-forward routing for
           intermittent connectivity scenarios.

        8. **Adaptive Link Strategy** - Dynamic modulation and power control based
           on channel conditions.

        ### 🎯 Getting Started

        1. Select a simulation from the sidebar
        2. Configure parameters using the control panel
        3. Click "Run Simulation" to execute
        4. Explore interactive visualizations
        5. Analyze performance metrics

        ### 💡 Learning Approach

        Each simulation provides:
        - **Interactive controls** for exploring parameter space
        - **Real-time visualizations** showing algorithm behavior
        - **Educational commentary** explaining key concepts
        - **Performance metrics** demonstrating trade-offs
        """)

    with col2:
        st.info("""
        **Quick Tips:**

        🎛️ Use the sidebar to navigate between simulations

        📊 Check statistics to track your progress

        ⚙️ Adjust configuration for different scenarios

        💾 All simulation runs are logged to the database
        """)

        st.success("""
        **Ready to begin?**

        Select a simulation from the sidebar to start exploring!
        """)

        # Recent activity
        st.subheader("Recent Activity")
        try:
            recent = st.session_state.db.fetch_recent_runs(limit=5)
            if recent:
                for run in recent:
                    success_icon = "✅" if run['success'] else "❌"
                    st.text(f"{success_icon} {run['simulation_type']}")
            else:
                st.text("No simulations run yet")
        except:
            st.text("Database not available")


def render_high_order_modulation():
    """Render High-Order Modulation simulation page."""
    from simulations.high_order_modulation import HighOrderModulationSimulation
    from visualizations.high_order_constellation import HighOrderConstellationPlot

    st.header("📡 High-Order Modulation (Civilization 1)")

    st.markdown("""
    **Alien Civilization 1** uses advanced constellation modulation to encode multiple bits
    per symbol. Explore how modulation order affects spectral efficiency and error rates.

    **Learning Objectives:**
    - Understand bit-to-symbol mapping
    - Visualize constellation diagrams
    - Observe noise effects on symbols
    - Analyze BER vs SNR trade-offs
    """)

    # Create simulation instance
    sim = HighOrderModulationSimulation()

    # Layout: controls on left, visualization on right
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Parameters")

        # Modulation order selection
        mod_order = st.selectbox(
            "Modulation Order",
            options=[4, 16, 64, 256],
            index=1,  # Default to 16-QAM
            help="Higher orders encode more bits/symbol but are less noise-resistant"
        )

        # Number of symbols
        num_symbols = st.slider(
            "Number of Symbols",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="More symbols provide better statistics but slower rendering"
        )

        # SNR control
        snr_db = st.slider(
            "SNR (dB)",
            min_value=-5.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
            help="Signal-to-Noise Ratio - higher means less noise"
        )

        # Modulation type
        mod_type = st.radio(
            "Modulation Type",
            options=['QAM', 'PSK'],
            index=0,
            help="QAM: Quadrature Amplitude Modulation\nPSK: Phase Shift Keying"
        )

        # Run button
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

        st.divider()

        # Educational info
        with st.expander("📚 About this Simulation"):
            bits_per_sym = int(np.log2(mod_order))
            st.write(f"**{mod_order}-{mod_type}** encodes **{bits_per_sym} bits** per symbol")
            st.write(f"**Spectral Efficiency:** {bits_per_sym} bits/s/Hz")
            st.write(f"**Constellation Size:** {mod_order} points")

            st.markdown("""
            **How it works:**
            1. Random bits are generated
            2. Bits are grouped (e.g., 4 bits for 16-QAM)
            3. Each group maps to a constellation point
            4. AWGN noise is added based on SNR
            5. Receiver detects closest constellation point
            """)

    with col2:
        st.subheader("📊 Constellation Diagram")

        # Run simulation if button clicked or if never run
        if run_button or 'last_mod_result' not in st.session_state:
            with st.spinner("Running simulation..."):
                # Prepare parameters
                params = {
                    'modulation_order': mod_order,
                    'num_symbols': num_symbols,
                    'snr_db': snr_db,
                    'modulation_type': mod_type,
                    'use_gray_coding': True
                }

                # Run simulation
                result = sim.run_simulation(params)

                # Store in session state
                st.session_state.last_mod_result = result

                # Log to database
                try:
                    sim_id = st.session_state.db.insert_simulation_run(
                        simulation_type="HighOrderModulation",
                        start_time=result.timestamp,
                        parameters=params,
                        simulation_name=f"{mod_order}-{mod_type}"
                    )
                    st.session_state.db.update_simulation_run(
                        sim_id=sim_id,
                        end_time=datetime.now(),
                        success=result.success,
                        execution_time_ms=result.execution_time_ms,
                        result_summary={
                            'ber': result.data.get('ber', 0),
                            'ser': result.data.get('ser', 0)
                        },
                        error_message=result.error_message
                    )
                except Exception as e:
                    st.warning(f"Could not log to database: {e}")

        # Display results
        if 'last_mod_result' in st.session_state:
            result = st.session_state.last_mod_result

            if result.success:
                # Create and render visualization
                viz = HighOrderConstellationPlot(backend='plotly')
                fig = viz.render(result)
                st.plotly_chart(fig, use_container_width=True)

                # Display metrics
                st.divider()
                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    st.metric(
                        "Bit Error Rate",
                        f"{result.data['ber']:.4f}",
                        help="Fraction of bits incorrectly decoded"
                    )

                with col_b:
                    st.metric(
                        "Symbol Error Rate",
                        f"{result.data['ser']:.4f}",
                        help="Fraction of symbols incorrectly decoded"
                    )

                with col_c:
                    st.metric(
                        "Bit Errors",
                        f"{result.data['num_bit_errors']}/{len(result.data['bits'])}",
                        help="Number of bit errors out of total bits"
                    )

                with col_d:
                    st.metric(
                        "Execution Time",
                        f"{result.execution_time_ms:.1f} ms",
                        help="Simulation computation time"
                    )

                # Analysis section
                with st.expander("📈 Performance Analysis"):
                    st.markdown(f"""
                    **Simulation Results for {mod_order}-{mod_type}:**

                    - **Modulation Order:** {mod_order} ({int(np.log2(mod_order))} bits/symbol)
                    - **SNR:** {snr_db:.1f} dB
                    - **Symbols Transmitted:** {num_symbols}
                    - **Total Bits:** {len(result.data['bits'])}

                    **Error Statistics:**
                    - **BER:** {result.data['ber']:.6f} ({result.data['ber']*100:.4f}%)
                    - **SER:** {result.data['ser']:.6f} ({result.data['ser']*100:.4f}%)
                    - **Bit Errors:** {result.data['num_bit_errors']}
                    - **Symbol Errors:** {result.data['num_symbol_errors']}

                    **Interpretation:**
                    {"✅ Excellent performance! Low error rate indicates good channel conditions." if result.data['ber'] < 0.01 else ""}
                    {"⚠️ Moderate errors. Consider increasing SNR or reducing modulation order." if 0.01 <= result.data['ber'] < 0.1 else ""}
                    {"❌ High error rate. Channel quality is poor for this modulation order." if result.data['ber'] >= 0.1 else ""}
                    """)

            else:
                st.error(f"Simulation failed: {result.error_message}")
        else:
            st.info("Click 'Run Simulation' to begin")


def render_ofdm_simulation():
    """Render OFDM Signal Processing simulation page."""
    from simulations.ofdm import OFDMSimulation
    from visualizations.ofdm_constellation import OFDMConstellationPlot

    st.header("📡 OFDM Signal Processing (Civilization 2)")

    st.markdown("""
    **Alien Civilization 2** uses Orthogonal Frequency Division Multiplexing to transmit
    data in parallel across multiple subcarriers. Explore how OFDM achieves high spectral
    efficiency and robustness against frequency-selective fading.

    **Learning Objectives:**
    - Understand IFFT/FFT for OFDM signal generation
    - Learn the purpose of cyclic prefix
    - Visualize parallel subcarrier transmission
    - Explore pilot-aided channel estimation
    """)

    # Create simulation instance
    sim = OFDMSimulation()

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Parameters")

        # Number of subcarriers
        num_subcarriers = st.selectbox(
            "Number of Subcarriers",
            options=[16, 32, 64, 128, 256],
            index=2,  # Default to 64
            help="FFT size - more subcarriers = finer frequency resolution"
        )

        # Cyclic prefix length
        cp_length = st.slider(
            "Cyclic Prefix Length",
            min_value=0,
            max_value=num_subcarriers // 4,
            value=num_subcarriers // 4,
            step=4,
            help="CP guards against intersymbol interference"
        )

        # Number of OFDM symbols
        num_symbols = st.slider(
            "OFDM Symbols",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of OFDM symbols to transmit"
        )

        # Subcarrier modulation
        mod_type = st.selectbox(
            "Subcarrier Modulation",
            options=['BPSK', 'QPSK', '16QAM'],
            index=1,
            help="Modulation applied to each subcarrier"
        )

        # SNR
        snr_db = st.slider(
            "SNR (dB)",
            min_value=0.0,
            max_value=35.0,
            value=15.0,
            step=1.0,
            help="Signal-to-Noise Ratio"
        )

        # Pilot spacing
        pilot_spacing = st.slider(
            "Pilot Spacing",
            min_value=0,
            max_value=16,
            value=8,
            step=4,
            help="0 = no pilots, otherwise insert pilot every N subcarriers"
        )

        # Run button
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

        st.divider()

        # Educational info
        with st.expander("📚 About OFDM"):
            bits_per_symbol = {'BPSK': 1, 'QPSK': 2, '16QAM': 4}[mod_type]
            num_pilots = num_subcarriers // pilot_spacing if pilot_spacing > 0 else 0
            num_data = num_subcarriers - num_pilots
            spectral_eff = bits_per_symbol * num_data / (num_subcarriers + cp_length)

            st.write(f"**Subcarriers:** {num_subcarriers}")
            st.write(f"**Data Subcarriers:** {num_data}")
            st.write(f"**Pilot Subcarriers:** {num_pilots}")
            st.write(f"**Spectral Efficiency:** {spectral_eff:.2f} bits/s/Hz")

            st.markdown("""
            **OFDM Advantages:**
            - High spectral efficiency
            - Robustness to multipath fading
            - Flexible resource allocation
            - Used in WiFi, LTE, 5G

            **Key Components:**
            - IFFT: Converts frequency to time domain
            - Cyclic Prefix: Eliminates ISI
            - Pilots: Enable channel estimation
            """)

    with col2:
        st.subheader("📊 OFDM Visualization")

        # Run simulation
        if run_button or 'last_ofdm_result' not in st.session_state:
            with st.spinner("Running OFDM simulation..."):
                params = {
                    'num_subcarriers': num_subcarriers,
                    'cp_length': cp_length,
                    'num_symbols': num_symbols,
                    'subcarrier_modulation': mod_type,
                    'snr_db': snr_db,
                    'pilot_spacing': pilot_spacing
                }

                result = sim.run_simulation(params)
                st.session_state.last_ofdm_result = result

                # Log to database
                try:
                    sim_id = st.session_state.db.insert_simulation_run(
                        simulation_type="OFDM",
                        start_time=result.timestamp,
                        parameters=params,
                        simulation_name=f"{num_subcarriers}-SC OFDM"
                    )
                    st.session_state.db.update_simulation_run(
                        sim_id=sim_id,
                        end_time=datetime.now(),
                        success=result.success,
                        execution_time_ms=result.execution_time_ms,
                        result_summary={'ber': result.data.get('ber', 0)},
                        error_message=result.error_message
                    )
                except Exception as e:
                    st.warning(f"Could not log to database: {e}")

        # Display results
        if 'last_ofdm_result' in st.session_state:
            result = st.session_state.last_ofdm_result

            if result.success:
                # Render visualization
                viz = OFDMConstellationPlot(backend='plotly')
                fig = viz.render(result)
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                st.divider()
                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    st.metric("BER", f"{result.data['ber']:.6f}")

                with col_b:
                    st.metric(
                        "Spectral Efficiency",
                        f"{result.metadata['spectral_efficiency']:.2f}",
                        help="Bits per second per Hz"
                    )

                with col_c:
                    st.metric(
                        "Bit Errors",
                        f"{result.data['num_bit_errors']}/{len(result.data['bits'])}"
                    )

                with col_d:
                    st.metric("Exec Time", f"{result.execution_time_ms:.1f} ms")

                # Analysis
                with st.expander("📈 OFDM Performance Analysis"):
                    st.markdown(f"""
                    **Configuration:**
                    - Subcarriers: {result.metadata['num_subcarriers']}
                    - CP Length: {result.metadata['cp_length']} samples
                    - Modulation: {result.metadata['subcarrier_modulation']}
                    - Bits/Symbol: {result.metadata['bits_per_symbol']}

                    **Results:**
                    - BER: {result.data['ber']:.8f}
                    - Bit Errors: {result.data['num_bit_errors']}
                    - SNR: {result.data['actual_snr_db']:.2f} dB
                    - Spectral Efficiency: {result.metadata['spectral_efficiency']:.3f} bits/s/Hz

                    **CP Overhead:** {result.metadata['cp_length'] / (result.metadata['num_subcarriers'] + result.metadata['cp_length']) * 100:.1f}%

                    {"✅ Excellent! OFDM performing well." if result.data['ber'] < 0.001 else ""}
                    {"⚠️ Moderate errors detected." if 0.001 <= result.data['ber'] < 0.01 else ""}
                    {"❌ High error rate - increase SNR or reduce modulation order." if result.data['ber'] >= 0.01 else ""}
                    """)
            else:
                st.error(f"Simulation failed: {result.error_message}")
        else:
            st.info("Click 'Run Simulation' to begin")


def render_placeholder_simulation(sim_name: str):
    """Render placeholder for simulation modules (to be implemented)."""
    st.header(f"📡 {sim_name}")

    st.info("""
    This simulation module is under development and will be available soon.

    **Planned features:**
    - Interactive parameter controls
    - Real-time animated visualizations
    - Performance metrics analysis
    - Educational commentary

    Check back after more modules are implemented!
    """)

    # Show where controls and visualization would go
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Parameters")
        st.text("Parameter controls will appear here")

        st.button("Run Simulation", disabled=True, help="Coming soon!")

    with col2:
        st.subheader("Visualization")
        st.text("Interactive visualization will appear here")


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Render UI components
    render_header()
    render_sidebar()

    # Main content area - route to appropriate page
    selected = st.session_state.selected_simulation

    if selected == "Welcome" or selected is None:
        render_welcome_page()
    elif selected == "High-Order Modulation (Civilization 1)":
        render_high_order_modulation()
    elif selected == "OFDM Signal Processing (Civilization 2)":
        render_ofdm_simulation()
    else:
        # Placeholder for other simulation modules (will be implemented later)
        render_placeholder_simulation(selected)

    # Footer
    st.divider()
    st.caption("Orbiter-2 v1.0.0-dev | Educational Wireless Communications Platform | Claude AI Implementation")


if __name__ == "__main__":
    main()
