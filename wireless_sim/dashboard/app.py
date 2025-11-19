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


def render_convolutional_coding():
    """Render Convolutional Coding simulation page."""
    from simulations.convolutional_coding import ConvolutionalCodingSimulation
    from visualizations.convolutional_trellis import ConvolutionalTrellisPlot
    import numpy as np

    st.header("📡 Convolutional Coding (Civilization 3)")

    st.markdown("""
    **Alien Civilization 3** employs convolutional codes with Viterbi decoding for robust
    forward error correction. Explore how trellis-based codes protect data against channel
    errors through memory-based encoding and maximum likelihood decoding.

    **Learning Objectives:**
    - Understand convolutional code structure
    - Visualize trellis state diagrams
    - Learn the Viterbi algorithm
    - Explore coding gain vs uncoded transmission
    """)

    # Create simulation instance
    sim = ConvolutionalCodingSimulation()

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Parameters")

        # Constraint length
        constraint_length = st.selectbox(
            "Constraint Length (K)",
            options=[3, 5, 7, 9],
            index=2,  # Default to K=7
            help="K=7 is industry standard (used in WiFi, satellites)"
        )

        # Number of information bits
        num_bits = st.slider(
            "Information Bits",
            min_value=20,
            max_value=200,
            value=50,
            step=10,
            help="Number of information bits to encode (keep small for trellis viz)"
        )

        # SNR
        snr_db = st.slider(
            "SNR (dB)",
            min_value=-2.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Lower SNR demonstrates coding gain better"
        )

        # Generator polynomials (show for reference)
        generator_polys = {
            3: [0o5, 0o7],
            5: [0o23, 0o35],
            7: [0o171, 0o133],
            9: [0o753, 0o561]
        }
        gen_poly = generator_polys[constraint_length]

        st.text(f"Generator polynomials (octal):")
        st.code(f"G1 = {oct(gen_poly[0])}\nG2 = {oct(gen_poly[1])}", language="text")

        # Code rate
        st.metric("Code Rate", "1/2", help="2 output bits per 1 input bit")

        # Run button
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

        st.divider()

        # Educational info
        with st.expander("📚 About Convolutional Coding"):
            num_states = 2 ** (constraint_length - 1)
            st.write(f"**Constraint Length:** K={constraint_length}")
            st.write(f"**Number of States:** {num_states}")
            st.write(f"**Memory Length:** {constraint_length - 1}")
            st.write(f"**Code Rate:** 1/2 (50% efficiency)")

            st.markdown("""
            **How it works:**
            1. Input bits shift through register
            2. Generator polynomials create output bits
            3. Rate 1/2 means 2 outputs per 1 input
            4. Viterbi decoder finds most likely path
            5. Soft decisions use received values

            **Applications:**
            - Satellite communications
            - Deep-space probes (Voyager, etc.)
            - WiFi (802.11)
            - GSM cellular
            - Digital video broadcasting
            """)

    with col2:
        st.subheader("📊 Trellis Diagram & Performance")

        # Run simulation
        if run_button or 'last_conv_result' not in st.session_state:
            with st.spinner("Running convolutional coding simulation..."):
                params = {
                    'constraint_length': constraint_length,
                    'num_bits': num_bits,
                    'snr_db': snr_db,
                    'generator_polynomials': gen_poly
                }

                result = sim.run_simulation(params)
                st.session_state.last_conv_result = result

                # Log to database
                try:
                    sim_id = st.session_state.db.insert_simulation_run(
                        simulation_type="ConvolutionalCoding",
                        start_time=result.timestamp,
                        parameters=params,
                        simulation_name=f"K={constraint_length} Conv Code"
                    )
                    st.session_state.db.update_simulation_run(
                        sim_id=sim_id,
                        end_time=datetime.now(),
                        success=result.success,
                        execution_time_ms=result.execution_time_ms,
                        result_summary={
                            'ber': result.data.get('ber', 0),
                            'coding_gain_db': result.data.get('coding_gain_db', 0)
                        },
                        error_message=result.error_message
                    )
                except Exception as e:
                    st.warning(f"Could not log to database: {e}")

        # Display results
        if 'last_conv_result' in st.session_state:
            result = st.session_state.last_conv_result

            if result.success:
                # Render trellis visualization
                viz = ConvolutionalTrellisPlot(backend='plotly', max_stages=20)
                fig = viz.render(result)
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                st.divider()
                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    st.metric(
                        "Coded BER",
                        f"{result.data['ber']:.6f}",
                        help="Bit error rate with convolutional coding"
                    )

                with col_b:
                    st.metric(
                        "Uncoded BER",
                        f"{result.data['uncoded_ber']:.6f}",
                        help="BER without coding (for comparison)"
                    )

                with col_c:
                    coding_gain = result.data['coding_gain_db']
                    st.metric(
                        "Coding Gain",
                        f"{coding_gain:.2f} dB" if coding_gain < float('inf') else "∞ dB",
                        help="Improvement over uncoded transmission"
                    )

                with col_d:
                    st.metric(
                        "Bit Errors",
                        f"{result.data['bit_errors']}/{num_bits}"
                    )

                # Analysis
                with st.expander("📈 Coding Performance Analysis"):
                    st.markdown(f"""
                    **Code Configuration:**
                    - Constraint Length: K={result.metadata['constraint_length']}
                    - Code Rate: {result.metadata['code_rate']}
                    - Number of States: {result.metadata['num_states']}
                    - Generator Polynomials: {[oct(g) for g in result.metadata['generator_polynomials']]}

                    **Performance Results:**
                    - **Coded BER:** {result.data['ber']:.8f}
                    - **Uncoded BER:** {result.data['uncoded_ber']:.8f}
                    - **Coding Gain:** {result.data['coding_gain_db']:.2f} dB
                    - **Bit Errors (Coded):** {result.data['bit_errors']}/{num_bits}
                    - **Bit Errors (Uncoded):** {result.data['uncoded_errors']}/{num_bits}

                    **SNR:** {snr_db:.1f} dB

                    **Interpretation:**
                    {"✅ Excellent coding gain! Convolutional code significantly reduced errors." if result.data['coding_gain_db'] > 3.0 else ""}
                    {"⚠️ Moderate coding gain. Consider higher constraint length." if 1.0 <= result.data['coding_gain_db'] <= 3.0 else ""}
                    {"📊 Minimal errors detected - SNR may be too high to see coding benefit." if result.data['ber'] < 0.001 and result.data['uncoded_ber'] < 0.001 else ""}

                    **Trellis Diagram Above:**
                    - Red path shows the decoded survivor path
                    - Each stage represents one information bit
                    - States shown as binary representations
                    - Viterbi algorithm finds maximum likelihood path
                    """)
            else:
                st.error(f"Simulation failed: {result.error_message}")
        else:
            st.info("Click 'Run Simulation' to begin")


def render_ldpc_decoding():
    """Render LDPC Decoding simulation page."""
    from simulations.ldpc_decoding import LDPCDecodingSimulation
    from visualizations.ldpc_iteration_heatmap import LDPCIterationHeatmap
    import numpy as np

    st.header("📡 LDPC Decoding (Civilization 4)")

    st.markdown("""
    **Alien Civilization 4** uses Low-Density Parity-Check (LDPC) codes with belief
    propagation for near-capacity error correction. Explore how iterative message passing
    on sparse graphs achieves excellent performance.

    **Learning Objectives:**
    - Understand sparse parity-check matrices
    - Learn belief propagation algorithm
    - Visualize LLR convergence across iterations
    - Explore near-Shannon-limit performance
    """)

    # Create simulation instance
    sim = LDPCDecodingSimulation()

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Parameters")

        # Block length
        block_length = st.selectbox(
            "Block Length (n)",
            options=[50, 100, 150, 200],
            index=1,  # Default to 100
            help="LDPC codeword length"
        )

        # Code rate
        code_rate = st.slider(
            "Code Rate",
            min_value=0.3,
            max_value=0.8,
            value=0.5,
            step=0.1,
            help="Information bits / codeword bits (k/n)"
        )

        # Max iterations
        max_iterations = st.slider(
            "Max BP Iterations",
            min_value=5,
            max_value=30,
            value=20,
            step=5,
            help="Maximum belief propagation iterations"
        )

        # SNR
        snr_db = st.slider(
            "SNR (dB)",
            min_value=-1.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Signal-to-Noise Ratio for BPSK"
        )

        # Early termination
        early_term = st.checkbox(
            "Early Termination",
            value=True,
            help="Stop iterations if syndrome becomes zero"
        )

        # Variable and check degrees
        st.text("Graph Structure:")
        col_a, col_b = st.columns(2)
        with col_a:
            var_degree = st.number_input("dv", min_value=2, max_value=5, value=3,
                                        help="Variable node degree")
        with col_b:
            check_degree = st.number_input("dc", min_value=4, max_value=10, value=6,
                                          help="Check node degree")

        # Info metrics
        k = int(block_length * code_rate)
        st.metric("Info Bits (k)", k)
        st.metric("Parity Bits (m)", block_length - k)

        # Run button
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

        st.divider()

        # Educational info
        with st.expander("📚 About LDPC Codes"):
            st.markdown("""
            **LDPC Codes** are graph-based codes that achieve
            near-Shannon-limit performance through iterative decoding.

            **Key Features:**
            - Sparse parity-check matrix H
            - Tanner graph representation
            - Belief propagation (sum-product algorithm)
            - Iterative soft-decision decoding

            **Applications:**
            - 5G NR (New Radio)
            - WiFi 6 (802.11ax)
            - DVB-S2 (satellite TV)
            - 10GBASE-T Ethernet
            - Deep space communications

            **Decoding Process:**
            1. Initialize variable nodes with channel LLRs
            2. Check nodes send parity messages
            3. Variable nodes update beliefs
            4. Repeat until convergence or max iterations
            """)

    with col2:
        st.subheader("📊 Iteration Heatmap & Convergence")

        # Run simulation
        if run_button or 'last_ldpc_result' not in st.session_state:
            with st.spinner("Running LDPC decoding simulation..."):
                params = {
                    'block_length': block_length,
                    'code_rate': code_rate,
                    'max_iterations': max_iterations,
                    'snr_db': snr_db,
                    'early_termination': early_term,
                    'variable_degree': var_degree,
                    'check_degree': check_degree
                }

                result = sim.run_simulation(params)
                st.session_state.last_ldpc_result = result

                # Log to database
                try:
                    sim_id = st.session_state.db.insert_simulation_run(
                        simulation_type="LDPC",
                        start_time=result.timestamp,
                        parameters=params,
                        simulation_name=f"n={block_length}, R={code_rate}"
                    )
                    st.session_state.db.update_simulation_run(
                        sim_id=sim_id,
                        end_time=datetime.now(),
                        success=result.success,
                        execution_time_ms=result.execution_time_ms,
                        result_summary={
                            'ber': result.data.get('ber', 0),
                            'converged': result.data.get('converged', False)
                        },
                        error_message=result.error_message
                    )
                except Exception as e:
                    st.warning(f"Could not log to database: {e}")

        # Display results
        if 'last_ldpc_result' in st.session_state:
            result = st.session_state.last_ldpc_result

            if result.success:
                # Render visualization
                viz = LDPCIterationHeatmap(backend='plotly')
                fig = viz.render(result)
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                st.divider()
                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    st.metric(
                        "BER",
                        f"{result.data['ber']:.6f}",
                        help="Bit error rate (all bits)"
                    )

                with col_b:
                    st.metric(
                        "Info BER",
                        f"{result.data['info_ber']:.6f}",
                        help="Bit error rate (info bits only)"
                    )

                with col_c:
                    converged_icon = "✓" if result.data['converged'] else "✗"
                    st.metric(
                        "Converged",
                        converged_icon,
                        help="Whether syndrome reached zero"
                    )

                with col_d:
                    st.metric(
                        "Iterations",
                        f"{result.data['num_iterations']}/{max_iterations}",
                        help="BP iterations performed"
                    )

                # Analysis
                with st.expander("📈 LDPC Performance Analysis"):
                    st.markdown(f"""
                    **Code Configuration:**
                    - Block Length (n): {result.metadata['block_length']}
                    - Info Length (k): {result.metadata['info_length']}
                    - Code Rate: {result.metadata['code_rate']:.2f}
                    - Parity Checks (m): {result.metadata['num_parity_checks']}
                    - Variable Node Degree: {result.metadata['variable_degree']}
                    - Check Node Degree: {result.metadata['check_degree']}

                    **Decoding Results:**
                    - **BER (all bits):** {result.data['ber']:.8f}
                    - **BER (info bits):** {result.data['info_ber']:.8f}
                    - **Bit Errors:** {result.data['bit_errors']}/{result.metadata['block_length']}
                    - **Info Bit Errors:** {result.data['info_bit_errors']}/{result.metadata['info_length']}
                    - **Converged:** {result.data['converged']}
                    - **Iterations Used:** {result.data['num_iterations']}/{result.metadata['max_iterations']}

                    **SNR:** {snr_db:.1f} dB

                    **Interpretation:**
                    {"✅ Perfect decoding! All bits correct, syndrome converged." if result.data['ber'] == 0 and result.data['converged'] else ""}
                    {"⚠️ Some errors remain. Try increasing SNR or max iterations." if result.data['ber'] > 0 and not result.data['converged'] else ""}
                    {"✓ Decoding converged (syndrome=0) with minimal errors." if result.data['converged'] and result.data['ber'] > 0 else ""}

                    **Belief Propagation:**
                    - LDPC uses iterative message passing on Tanner graph
                    - Variable nodes and check nodes exchange LLR messages
                    - Convergence indicated by syndrome weight reaching zero
                    - Each iteration refines bit confidence (LLR magnitudes)
                    - See heatmap above for LLR evolution
                    """)
            else:
                st.error(f"Simulation failed: {result.error_message}")
        else:
            st.info("Click 'Run Simulation' to begin")


def render_mesh_routing():
    """Render Mesh Routing simulation page."""
    from simulations.mesh_routing import MeshRoutingSimulation
    from visualizations.mesh_network_graph import MeshNetworkGraph
    import numpy as np

    st.header("📡 Mesh Routing (Civilization 5)")

    st.markdown("""
    **Alien Civilization 5** uses wireless mesh networks with multi-hop routing for
    decentralized communication. Explore how ad-hoc networks discover routes and
    forward packets without centralized infrastructure.

    **Learning Objectives:**
    - Understand wireless mesh network topologies
    - Learn route discovery and path selection
    - Visualize multi-hop packet forwarding
    - Analyze network connectivity and resilience
    """)

    # Create simulation instance
    sim = MeshRoutingSimulation()

    # Layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ Parameters")

        # Number of nodes
        num_nodes = st.slider(
            "Number of Nodes",
            min_value=5,
            max_value=30,
            value=15,
            help="Mesh network size"
        )

        # Network area
        network_area = st.slider(
            "Network Area (m)",
            min_value=50.0,
            max_value=150.0,
            value=100.0,
            step=10.0,
            help="Square network area side length"
        )

        # Transmission range
        tx_range = st.slider(
            "Transmission Range (m)",
            min_value=15.0,
            max_value=60.0,
            value=30.0,
            step=5.0,
            help="Radio transmission range"
        )

        # Number of packets
        num_packets = st.slider(
            "Packets to Route",
            min_value=5,
            max_value=25,
            value=10,
            help="Number of packets to simulate"
        )

        # Node failure probability
        failure_prob = st.slider(
            "Node Failure Probability",
            min_value=0.0,
            max_value=0.3,
            value=0.0,
            step=0.05,
            help="Probability of node being unavailable"
        )

        # Run button
        run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

        st.divider()

        # Educational info
        with st.expander("📚 About Mesh Networks"):
            st.markdown("""
            **Mesh Networks** enable devices to communicate
            without fixed infrastructure through multi-hop routing.

            **Key Features:**
            - Decentralized, self-organizing
            - Redundant paths improve reliability
            - Dynamic route discovery
            - Scalable topology

            **Routing:**
            - Uses shortest-path algorithms
            - Hop count as routing metric
            - Route maintenance and recovery

            **Applications:**
            - Smart city IoT networks
            - Disaster recovery communications
            - Military tactical networks
            - Community wireless networks
            """)

    with col2:
        st.subheader("📊 Network Topology")

        # Run simulation
        if run_button or 'last_mesh_result' not in st.session_state:
            with st.spinner("Simulating mesh network..."):
                params = {
                    'num_nodes': num_nodes,
                    'network_area': network_area,
                    'transmission_range': tx_range,
                    'num_packets': num_packets,
                    'node_failure_prob': failure_prob
                }

                result = sim.run_simulation(params)
                st.session_state.last_mesh_result = result

                # Log to database
                try:
                    sim_id = st.session_state.db.insert_simulation_run(
                        simulation_type="MeshRouting",
                        start_time=result.timestamp,
                        parameters=params,
                        simulation_name=f"{num_nodes} nodes"
                    )
                    st.session_state.db.update_simulation_run(
                        sim_id=sim_id,
                        end_time=datetime.now(),
                        success=result.success,
                        execution_time_ms=result.execution_time_ms,
                        result_summary={
                            'pdr': result.data.get('packet_delivery_ratio', 0),
                            'connected': result.data.get('is_connected', False)
                        },
                        error_message=result.error_message
                    )
                except Exception as e:
                    st.warning(f"Could not log to database: {e}")

        # Display results
        if 'last_mesh_result' in st.session_state:
            result = st.session_state.last_mesh_result

            if result.success:
                # Render visualization
                viz = MeshNetworkGraph(backend='plotly')
                fig = viz.render(result)
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                st.divider()
                col_a, col_b, col_c, col_d = st.columns(4)

                with col_a:
                    st.metric(
                        "PDR",
                        f"{result.data['packet_delivery_ratio']:.1%}",
                        help="Packet Delivery Ratio"
                    )

                with col_b:
                    connected_icon = "✓" if result.data['is_connected'] else "✗"
                    st.metric(
                        "Connected",
                        connected_icon,
                        help="Network fully connected"
                    )

                with col_c:
                    st.metric(
                        "Avg Hops",
                        f"{result.data['avg_hop_count']:.1f}",
                        help="Average hop count"
                    )

                with col_d:
                    st.metric(
                        "Avg Latency",
                        f"{result.data['avg_latency_ms']:.1f} ms"
                    )

                # Analysis
                with st.expander("📈 Network Analysis"):
                    st.markdown(f"""
                    **Network Topology:**
                    - **Nodes:** {result.metadata['num_nodes']}
                    - **Edges:** {result.data['num_edges']}
                    - **Avg Node Degree:** {result.data['avg_node_degree']:.2f}
                    - **Connected:** {result.data['is_connected']}

                    **Routing Performance:**
                    - **Packet Delivery Ratio:** {result.data['packet_delivery_ratio']:.1%}
                    - **Successful Packets:** {result.data['successful_packets']}/{result.metadata['num_packets']}
                    - **Failed Packets:** {result.data['failed_packets']}
                    - **Avg Hop Count:** {result.data['avg_hop_count']:.2f}
                    - **Avg Latency:** {result.data['avg_latency_ms']:.2f} ms

                    **Interpretation:**
                    {"✅ Excellent! Network is fully connected and all packets delivered." if result.data['is_connected'] and result.data['packet_delivery_ratio'] == 1.0 else ""}
                    {"⚠️ Network is disconnected - some nodes cannot reach others." if not result.data['is_connected'] else ""}
                    {"⚠️ Some packet delivery failures. Increase transmission range or node density." if result.data['packet_delivery_ratio'] < 1.0 else ""}
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
    elif selected == "Convolutional Coding (Civilization 3)":
        render_convolutional_coding()
    elif selected == "LDPC Decoding (Civilization 4)":
        render_ldpc_decoding()
    elif selected == "Mesh Routing (Civilization 5)":
        render_mesh_routing()
    else:
        # Placeholder for other simulation modules (will be implemented later)
        render_placeholder_simulation(selected)

    # Footer
    st.divider()
    st.caption("Orbiter-2 v1.0.0-dev | Educational Wireless Communications Platform | Claude AI Implementation")


if __name__ == "__main__":
    main()
