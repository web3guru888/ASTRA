#!/bin/bash
# ============================================================
# Athena++ 3D Filament Simulation - Run Script
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================
# CONFIGURATION
# ============================================================

ATHENA_DIR=""  # Path to Athena++ directory (set below)
N_CORES=16     # Number of CPU cores to use
RUN_MODE="interactive"  # Options: interactive, background, batch

# ============================================================
# FUNCTIONS
# ============================================================

print_header() {
    echo ""
    echo "============================================================"
    echo "  ATHENA++ 3D FILAMENT FRAGMENTATION SIMULATION"
    echo "============================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_dependencies() {
    echo "Checking dependencies..."

    # Check for HDF5
    if command -v h5dump &> /dev/null; then
        print_success "HDF5 installed"
    else
        print_error "HDF5 not found. Install with: brew install hdf5"
        exit 1
    fi

    # Check for MPI
    if command -v mpirun &> /dev/null; then
        print_success "MPI installed"
    else
        print_warning "MPI not found. Will run in single-core mode (NOT recommended)"
        N_CORES=1
    fi

    # Check for Python
    if command -v python3 &> /dev/null; then
        print_success "Python installed"
    else
        print_error "Python not found. Required for analysis."
        exit 1
    fi

    echo ""
}

setup_athena() {
    echo "Setting up Athena++..."

    if [ -z "$ATHENA_DIR" ]; then
        # Try to find Athena++ in common locations
        if [ -d "../../athena++" ]; then
            ATHENA_DIR="../../athena++"
        elif [ -d "~/athena++" ]; then
            ATHENA_DIR="~/athena++"
        elif [ -d "/opt/athena++" ]; then
            ATHENA_DIR="/opt/athena++"
        else
            print_error "Athena++ directory not found!"
            echo "Please edit this script and set ATHENA_DIR variable."
            exit 1
        fi
    fi

    # Expand tilde
    ATHENA_DIR="${ATHENA_DIR/#\~/$HOME}"

    echo "Using Athena++ directory: $ATHENA_DIR"
    cd "$ATHENA_DIR"

    # Copy problem generator
    print_success "Copied problem generator"
    cp ../filament.cpp src/pmods/

    echo ""
}

compile_athena() {
    echo "Compiling Athena++..."

    # Clean previous build
    make clean > /dev/null 2>&1

    # Configure
    echo "Running ./configure..."
    ./configure \
        --hdf5=$(brew --prefix hdf5 2>/dev/null || echo "/usr/local") \
        --mpi \
        --prob=filament_fragmentation \
        --coord=cartesian \
        --eos=iso_hydro \
        --flux=hlld

    if [ $? -ne 0 ]; then
        print_error "Configuration failed!"
        exit 1
    fi

    # Compile
    echo "Compiling (this may take a few minutes)..."
    make -j8

    if [ $? -ne 0 ]; then
        print_error "Compilation failed!"
        exit 1
    fi

    print_success "Compilation complete!"
    echo ""
}

run_simulation() {
    echo "Starting simulation..."
    echo "  Configuration: filament_3d_athinput.athdf"
    echo "  CPU cores: $N_CORES"
    echo "  Estimated runtime: 25-35 hours"
    echo ""

    # Copy input file to Athena directory
    cp ../filament_3d_athinput.athdf .

    # Create output directory
    mkdir -p output

    # Run based on mode
    case $RUN_MODE in
        "background")
            print_success "Starting in background mode..."
            nohup mpirun -np $N_CORES ./bin/athena++ \
                -i filament_3d_athinput.athdf \
                > simulation.log 2>&1 &

            echo ""
            echo "Simulation running in background!"
            echo "Monitor progress: tail -f simulation.log"
            echo "Stop simulation: pkill -f athena++"
            ;;
        "batch")
            print_success "Starting in batch mode..."
            # For SLURM or other batch schedulers
            echo "NOTE: Batch mode requires modification for your system"
            echo "Edit this script to add your batch submission command"
            ;;
        *)
            print_success "Starting in interactive mode..."
            echo "Press Ctrl+C to stop the simulation"
            echo ""

            mpirun -np $N_CORES ./bin/athena++ \
                -i filament_3d_athinput.athdf
            ;;
    esac

    echo ""
}

analyze_results() {
    echo "Analyzing results..."

    # Go back to original directory
    cd -

    # Check if Python dependencies are installed
    python3 -c "import h5py, numpy, matplotlib, scipy" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "Installing Python dependencies..."
        pip3 install h5py numpy matplotlib scipy
    fi

    # Run analysis
    python3 analyze_filament.py

    print_success "Analysis complete!"
    echo ""
}

# ============================================================
# MAIN SCRIPT
# ============================================================

main() {
    print_header

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--compile)
                check_dependencies
                setup_athena
                compile_athena
                shift
                ;;
            -r|--run)
                run_simulation
                shift
                ;;
            -a|--analyze)
                analyze_results
                shift
                ;;
            -b|--background)
                RUN_MODE="background"
                shift
                ;;
            --cores)
                N_CORES=$2
                shift 2
                ;;
            --athena-dir)
                ATHENA_DIR=$2
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -c, --compile      Compile Athena++ with filament problem"
                echo "  -r, --run          Run the simulation"
                echo "  -a, --analyze     Analyze simulation results"
                echo "  -b, --background   Run in background mode"
                echo "  --cores N          Number of CPU cores (default: 16)"
                echo "  --athena-dir PATH  Path to Athena++ directory"
                echo "  -h, --help         Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0 --compile                    # Compile only"
                echo "  $0 --run --background --cores 32  # Run in background with 32 cores"
                echo "  $0 --analyze                    # Analyze results"
                echo ""
                echo "Full workflow:"
                echo "  $0 --compile && $0 --run && $0 --analyze"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # If no arguments provided, show help
    if [ $# -eq 0 ]; then
        print_warning "No arguments provided"
        exec $0 --help
    fi
}

main "$@"
