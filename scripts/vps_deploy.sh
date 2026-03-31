#!/bin/bash
# BioPAT Benchmark - VPS Deployment Script
# =========================================
# This script sets up and runs the BioPAT benchmark pipeline on a VPS.
#
# Usage:
#   chmod +x scripts/vps_deploy.sh
#   ./scripts/vps_deploy.sh [--skip-setup] [--skip-baseline]
#
# Environment variables (set these before running or create .env file):
#   PATENTSVIEW_API_KEYS  - Comma-separated PatentsView API keys (optional but recommended)
#   OPENALEX_MAILTO       - Email for OpenAlex polite pool (recommended)
#   EPO_CONSUMER_KEY      - EPO OPS consumer key (for Phase 6)
#   EPO_CONSUMER_SECRET   - EPO OPS consumer secret (for Phase 6)

set -e  # Exit on error

# Configuration
PYTHON_VERSION="3.12"
PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/venv}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/logs/pipeline_$(date +%Y%m%d_%H%M%S).log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

# Parse arguments
SKIP_SETUP=false
SKIP_BASELINE=false
PHASE="phase1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --phase)
            PHASE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-setup     Skip system setup and dependency installation"
            echo "  --skip-baseline  Skip BM25 baseline evaluation"
            echo "  --phase PHASE    Pipeline phase to run (phase1, phase2, phase3)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

log "=========================================="
log "BioPAT Benchmark - VPS Deployment"
log "=========================================="
log "Project directory: $PROJECT_DIR"
log "Data directory: $DATA_DIR"
log "Log file: $LOG_FILE"
log "Phase: $PHASE"

# ============================================
# System Setup (Ubuntu/Debian)
# ============================================
setup_system() {
    log "Setting up system dependencies..."

    # Detect OS
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
    else
        OS=$(uname -s)
    fi

    case $OS in
        ubuntu|debian)
            log "Detected Ubuntu/Debian"
            sudo apt-get update
            sudo apt-get install -y \
                python${PYTHON_VERSION} \
                python${PYTHON_VERSION}-venv \
                python${PYTHON_VERSION}-dev \
                build-essential \
                git \
                curl \
                wget \
                htop \
                tmux \
                libxml2-dev \
                libxslt1-dev \
                zlib1g-dev
            ;;
        centos|rhel|fedora)
            log "Detected CentOS/RHEL/Fedora"
            sudo dnf install -y \
                python${PYTHON_VERSION} \
                python${PYTHON_VERSION}-devel \
                gcc \
                gcc-c++ \
                make \
                git \
                curl \
                wget \
                htop \
                tmux \
                libxml2-devel \
                libxslt-devel \
                zlib-devel
            ;;
        Darwin)
            log "Detected macOS"
            if ! command -v brew &> /dev/null; then
                warn "Homebrew not found. Please install manually."
            else
                brew install python@${PYTHON_VERSION} git curl wget htop tmux
            fi
            ;;
        *)
            warn "Unknown OS: $OS. Skipping system package installation."
            ;;
    esac
}

# ============================================
# Python Environment Setup
# ============================================
setup_python() {
    log "Setting up Python virtual environment..."

    cd "$PROJECT_DIR"

    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment at $VENV_DIR"
        python${PYTHON_VERSION} -m venv "$VENV_DIR"
    fi

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    log "Upgrading pip..."
    pip install --upgrade pip wheel setuptools

    # Install project dependencies
    log "Installing project dependencies..."
    pip install -e ".[dev]"

    # Install evaluation dependencies (optional, for baseline)
    if [ "$SKIP_BASELINE" = false ]; then
        log "Installing evaluation dependencies..."
        pip install -e ".[evaluation]" || warn "Some evaluation deps failed (torch/faiss). BM25 should still work."
    fi

    log "Python environment ready."
}

# ============================================
# Configuration Setup
# ============================================
setup_config() {
    log "Setting up configuration..."

    mkdir -p "$PROJECT_DIR/configs"

    # Create default config if it doesn't exist
    if [ ! -f "$PROJECT_DIR/configs/default.yaml" ]; then
        log "Creating default configuration file..."
        cat > "$PROJECT_DIR/configs/default.yaml" << 'EOF'
# BioPAT Benchmark Configuration
# ================================
# Copy this file and modify for your environment

phase: "phase1"

paths:
  data_dir: "data"
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  cache_dir: "data/cache"
  benchmark_dir: "data/benchmark"

api:
  # PatentsView API keys (comma-separated for rotation)
  # Get keys at: https://patentsview.org/apis/keyrequest
  patentsview_api_keys: []

  # OpenAlex email for polite pool (faster rate limits)
  openalex_email: null

  # Use bulk data downloads where available
  use_bulk_data: true

phase1:
  # Target number of unique patents
  target_patent_count: 2000

  # Target number of queries (claims)
  target_query_count: 5000

  # Minimum citations per patent to include
  min_citations: 3

  # Reliance on Science confidence threshold (1-10)
  ros_confidence_threshold: 8

  # IPC prefixes for biomedical filtering
  ipc_prefixes:
    - "A61"  # Medical/veterinary science
    - "C07"  # Organic chemistry
    - "C12"  # Biochemistry

  # Random seed for reproducibility
  seed: 42
EOF
        log "Created configs/default.yaml"
    fi

    # Create .env file template if it doesn't exist
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log "Creating .env template..."
        cat > "$PROJECT_DIR/.env" << 'EOF'
# BioPAT Environment Variables
# ============================
# Fill in your API keys below

# PatentsView API keys (comma-separated for key rotation)
# Request at: https://patentsview.org/apis/keyrequest
PATENTSVIEW_API_KEYS=

# OpenAlex email for polite pool (recommended)
OPENALEX_MAILTO=

# EPO OPS credentials (for Phase 6 international patents)
# Register at: https://developers.epo.org/
EPO_CONSUMER_KEY=
EPO_CONSUMER_SECRET=
EOF
        log "Created .env template - please fill in your API keys"
    fi

    # Create data directories
    log "Creating data directories..."
    mkdir -p "$DATA_DIR"/{raw,processed,cache,benchmark}
}

# ============================================
# Run Pipeline
# ============================================
run_pipeline() {
    log "Starting BioPAT pipeline..."

    cd "$PROJECT_DIR"
    source "$VENV_DIR/bin/activate"

    # Load environment variables
    if [ -f "$PROJECT_DIR/.env" ]; then
        set -a
        source "$PROJECT_DIR/.env"
        set +a
    fi

    # Build command arguments
    ARGS="--config configs/default.yaml"

    if [ "$SKIP_BASELINE" = true ]; then
        ARGS="$ARGS --skip-baseline"
    fi

    # Run the appropriate phase
    case $PHASE in
        phase1)
            log "Running Phase 1 pipeline..."
            python -m biopat.pipeline $ARGS
            ;;
        phase2)
            log "Running Phase 2 pipeline..."
            python -m biopat.pipeline_phase2 $ARGS
            ;;
        phase3)
            log "Running Phase 3 pipeline..."
            python -m biopat.pipeline_phase3 $ARGS
            ;;
        *)
            error "Unknown phase: $PHASE"
            ;;
    esac

    log "Pipeline completed successfully!"
}

# ============================================
# Generate Summary Report
# ============================================
generate_report() {
    log "Generating summary report..."

    REPORT_FILE="$PROJECT_DIR/logs/report_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$REPORT_FILE" << EOF
BioPAT Benchmark - Run Report
=============================
Date: $(date)
Phase: $PHASE

Data Statistics
---------------
EOF

    # Count files in each directory
    echo "Raw data files: $(find "$DATA_DIR/raw" -type f 2>/dev/null | wc -l)" >> "$REPORT_FILE"
    echo "Processed files: $(find "$DATA_DIR/processed" -type f 2>/dev/null | wc -l)" >> "$REPORT_FILE"
    echo "Cache files: $(find "$DATA_DIR/cache" -type f 2>/dev/null | wc -l)" >> "$REPORT_FILE"
    echo "Benchmark files: $(find "$DATA_DIR/benchmark" -type f 2>/dev/null | wc -l)" >> "$REPORT_FILE"

    # Check for BEIR format outputs
    if [ -d "$DATA_DIR/benchmark" ]; then
        echo "" >> "$REPORT_FILE"
        echo "BEIR Output Structure" >> "$REPORT_FILE"
        echo "---------------------" >> "$REPORT_FILE"
        find "$DATA_DIR/benchmark" -name "*.jsonl" -o -name "*.tsv" 2>/dev/null | head -20 >> "$REPORT_FILE"
    fi

    # Disk usage
    echo "" >> "$REPORT_FILE"
    echo "Disk Usage" >> "$REPORT_FILE"
    echo "----------" >> "$REPORT_FILE"
    du -sh "$DATA_DIR"/* 2>/dev/null >> "$REPORT_FILE" || echo "N/A" >> "$REPORT_FILE"

    log "Report saved to: $REPORT_FILE"
    cat "$REPORT_FILE"
}

# ============================================
# Main Execution
# ============================================
main() {
    log "Starting deployment..."

    if [ "$SKIP_SETUP" = false ]; then
        setup_system
        setup_python
        setup_config
    else
        log "Skipping setup (--skip-setup flag)"
        # Still activate venv
        source "$VENV_DIR/bin/activate"
    fi

    run_pipeline
    generate_report

    log "=========================================="
    log "Deployment complete!"
    log "=========================================="
    log ""
    log "Next steps:"
    log "  1. Check the benchmark output in: $DATA_DIR/benchmark/"
    log "  2. Review the log file: $LOG_FILE"
    log "  3. Run evaluation: python -m biopat.evaluation.bm25"
    log ""
    log "To run in background with tmux:"
    log "  tmux new -s biopat './scripts/vps_deploy.sh --skip-setup'"
}

# Run main
main
