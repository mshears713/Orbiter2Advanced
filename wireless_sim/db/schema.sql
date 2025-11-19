-- Orbiter-2 Wireless Communications Simulation Database Schema
--
-- This schema defines tables for storing simulation metadata, run history,
-- and logging information. Designed for educational use and analysis of
-- simulation performance across different wireless communication techniques.
--
-- Author: Claude (Orbiter-2 AI Implementation)
-- Date: 2025-11-19

-- ============================================================================
-- Table: simulation_runs
-- ============================================================================
-- Stores metadata for each simulation execution including parameters,
-- timestamps, and success indicators.

CREATE TABLE IF NOT EXISTS simulation_runs (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Simulation identification
    simulation_type VARCHAR(100) NOT NULL,  -- e.g., "HighOrderModulation", "OFDM", etc.
    simulation_name VARCHAR(255),           -- Optional human-readable name

    -- Timing information
    start_time TIMESTAMP NOT NULL,          -- When simulation began
    end_time TIMESTAMP,                     -- When simulation completed (NULL if running)
    execution_time_ms REAL,                 -- Execution duration in milliseconds

    -- Parameters and results
    parameters TEXT NOT NULL,               -- JSON-encoded parameter dictionary
    result_summary TEXT,                    -- JSON-encoded summary of results

    -- Status tracking
    success BOOLEAN NOT NULL DEFAULT 0,     -- 1 if completed successfully, 0 otherwise
    error_message TEXT,                     -- Error description if failed

    -- Metadata
    user_notes TEXT,                        -- Optional notes from user
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexing for common queries
    CHECK (success IN (0, 1))
);

-- Index for querying by simulation type
CREATE INDEX IF NOT EXISTS idx_simulation_type
ON simulation_runs(simulation_type);

-- Index for querying by time range
CREATE INDEX IF NOT EXISTS idx_start_time
ON simulation_runs(start_time);

-- Index for querying successful runs
CREATE INDEX IF NOT EXISTS idx_success
ON simulation_runs(success);


-- ============================================================================
-- Table: logs
-- ============================================================================
-- Stores log messages generated during simulation execution for debugging
-- and analysis purposes.

CREATE TABLE IF NOT EXISTS logs (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Foreign key to simulation_runs
    sim_run_id INTEGER,

    -- Log entry details
    log_level VARCHAR(20) NOT NULL,         -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    message TEXT NOT NULL,                  -- Log message content
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Optional context
    module_name VARCHAR(100),               -- Which module generated the log
    function_name VARCHAR(100),             -- Which function generated the log
    line_number INTEGER,                    -- Line number in source code

    -- Additional data
    extra_data TEXT,                        -- JSON-encoded additional context

    -- Constraints
    FOREIGN KEY (sim_run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE,
    CHECK (log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'))
);

-- Index for querying logs by simulation run
CREATE INDEX IF NOT EXISTS idx_sim_run_id
ON logs(sim_run_id);

-- Index for querying logs by level
CREATE INDEX IF NOT EXISTS idx_log_level
ON logs(log_level);

-- Index for querying logs by timestamp
CREATE INDEX IF NOT EXISTS idx_log_timestamp
ON logs(timestamp);


-- ============================================================================
-- Table: simulation_metrics
-- ============================================================================
-- Stores performance metrics for simulations (BER, throughput, latency, etc.)

CREATE TABLE IF NOT EXISTS simulation_metrics (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Foreign key to simulation_runs
    sim_run_id INTEGER NOT NULL,

    -- Metric details
    metric_name VARCHAR(100) NOT NULL,      -- e.g., "BER", "SNR", "throughput"
    metric_value REAL NOT NULL,             -- Numeric value of the metric
    metric_unit VARCHAR(50),                -- Unit of measurement

    -- Timestamp
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    FOREIGN KEY (sim_run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE
);

-- Index for querying metrics by simulation run
CREATE INDEX IF NOT EXISTS idx_metrics_sim_run
ON simulation_metrics(sim_run_id);

-- Index for querying by metric name
CREATE INDEX IF NOT EXISTS idx_metric_name
ON simulation_metrics(metric_name);


-- ============================================================================
-- Table: configuration_history
-- ============================================================================
-- Tracks configuration changes over time for reproducibility

CREATE TABLE IF NOT EXISTS configuration_history (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Configuration details
    config_name VARCHAR(100) NOT NULL,      -- Name of configuration set
    config_data TEXT NOT NULL,              -- JSON-encoded configuration

    -- Versioning
    version INTEGER NOT NULL DEFAULT 1,
    is_active BOOLEAN DEFAULT 1,            -- Current active configuration

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CHECK (is_active IN (0, 1))
);

-- Index for querying active configurations
CREATE INDEX IF NOT EXISTS idx_active_config
ON configuration_history(is_active);


-- ============================================================================
-- Table: visualization_cache
-- ============================================================================
-- Caches visualization data for performance optimization

CREATE TABLE IF NOT EXISTS visualization_cache (
    -- Primary key
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Foreign key to simulation_runs
    sim_run_id INTEGER NOT NULL,

    -- Cache details
    viz_type VARCHAR(100) NOT NULL,         -- Type of visualization
    cached_data BLOB,                       -- Serialized visualization data
    cache_size_bytes INTEGER,               -- Size of cached data

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,

    -- Constraints
    FOREIGN KEY (sim_run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE
);

-- Index for querying cache by simulation run
CREATE INDEX IF NOT EXISTS idx_cache_sim_run
ON visualization_cache(sim_run_id);

-- Index for querying cache by visualization type
CREATE INDEX IF NOT EXISTS idx_viz_type
ON visualization_cache(viz_type);


-- ============================================================================
-- Views for common queries
-- ============================================================================

-- View: Recent successful simulations
CREATE VIEW IF NOT EXISTS recent_successful_runs AS
SELECT
    id,
    simulation_type,
    simulation_name,
    start_time,
    execution_time_ms,
    parameters
FROM simulation_runs
WHERE success = 1
ORDER BY start_time DESC
LIMIT 100;

-- View: Error summary
CREATE VIEW IF NOT EXISTS error_summary AS
SELECT
    simulation_type,
    COUNT(*) as error_count,
    MAX(start_time) as last_error_time
FROM simulation_runs
WHERE success = 0
GROUP BY simulation_type;

-- View: Average execution times by simulation type
CREATE VIEW IF NOT EXISTS avg_execution_times AS
SELECT
    simulation_type,
    COUNT(*) as run_count,
    AVG(execution_time_ms) as avg_time_ms,
    MIN(execution_time_ms) as min_time_ms,
    MAX(execution_time_ms) as max_time_ms
FROM simulation_runs
WHERE success = 1 AND execution_time_ms IS NOT NULL
GROUP BY simulation_type;


-- ============================================================================
-- Trigger: Update updated_at timestamp on configuration changes
-- ============================================================================

CREATE TRIGGER IF NOT EXISTS update_config_timestamp
AFTER UPDATE ON configuration_history
FOR EACH ROW
BEGIN
    UPDATE configuration_history
    SET updated_at = CURRENT_TIMESTAMP
    WHERE id = NEW.id;
END;


-- ============================================================================
-- Initial data: Insert default configuration
-- ============================================================================

INSERT OR IGNORE INTO configuration_history (id, config_name, config_data, version, is_active)
VALUES (
    1,
    'default',
    '{"simulation": {"sampling_rate": 1000000.0, "modulation_order": 16}, "database": {"enable_logging": true}}',
    1,
    1
);
