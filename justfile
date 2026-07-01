# =============================================================================
# justfile — DRLScheduler Pipeline Commands
# =============================================================================
# Usage: just <target>
# Example: just run_smoke
# Example: just run_full TRACE=deeplearn_job
# Ref: https://just.systems/man/en/
# =============================================================================

# Auto-detect CPU count (varies by OS)
cpu_count := if os() == "linux" {
    `nproc`
} else if os() == "macos" {
    `sysctl -n hw.ncpu`
} else {
    "4"
}

# Trace name override (just run_full TRACE=deeplearn_job)
TRACE := env_var_or_default("TRACE", "physical_job")

# Where `archive_results` copies analysis outputs + winning models off scratch
# into safe (home) storage. Override: just archive_results ARCHIVE=/path
ARCHIVE := env_var_or_default("ARCHIVE", env_var("HOME") + "/drl_archive")

# =============================================================================
# HELP
# =============================================================================

@help:
    echo "DRLScheduler Snakemake Pipeline — justfile Targets"
    echo ""
    echo "PIPELINE TARGETS:"
    echo "  dry_run_smoke        - Validate smoke DAG without execution"
    echo "  dry_run              - Validate production DAG without execution"
    echo "  run_smoke            - Smoke test (fast end-to-end validation)"
    echo "  run_full             - Full production pipeline (train → eval → aggregate → stats)"
    echo "  run_full_with_base   - Full pipeline + baseline comparison"
    echo "  run_baseline         - Run baseline scheduler only"
    echo ""
    echo "DAG EXPORT TARGETS:"
    echo "  export_dag           - Export both detail + overview DAGs"
    echo "  export_dag_detail    - Export job-level DAG (detailed)"
    echo "  export_dag_overview  - Export rule-level DAG (clean)"
    echo ""
    echo "SLURM TARGETS:"
    echo "  dry_run_smoke_slurm  - Validate smoke DAG for cluster"
    echo "  dry_run_slurm        - Validate production DAG for cluster"
    echo "  run_smoke_slurm      - Submit smoke test to SLURM"
    echo "  run_full_slurm       - Submit full pipeline to SLURM"
    echo "  run_full_with_base_slurm - Submit full pipeline + baseline to SLURM"
    echo "  archive_results      - Copy results + winning models off scratch to \$HOME/drl_archive"
    echo "  slurm_report         - Generate efficiency report after run"
    echo "  build_sif            - Build Apptainer .sif from Nix flake"
    echo ""
    echo "MAINTENANCE:"
    echo "  clean                - Remove all outputs except data and logs"
    echo "  clean_all            - Remove all outputs including logs"
    echo "  nix_develop          - Enter Nix shell"
    echo ""
    echo "EXAMPLES:"
    echo "  just run_smoke                         # Quick smoke test on physical_job"
    echo "  just run_full                          # Full run on physical_job"
    echo "  just run_full TRACE=deeplearn_job      # Full run on deeplearn_job"
    echo "  just export_dag                        # Export DAGs before running"
    echo "  just dry_run_smoke                     # Check DAG resolves correctly"
    echo ""

# =============================================================================
# VALIDATION TARGETS
# =============================================================================

@dry_run_smoke:
    echo "Validating smoke DAG (no execution)..."
    snakemake --configfile config.smoke.yaml --dry-run --quiet

@dry_run:
    echo "Validating production DAG (no execution)..."
    snakemake --configfile config.yaml --dry-run --quiet

# =============================================================================
# PIPELINE TARGETS
# =============================================================================

@run_smoke:
    echo "Running smoke test pipeline on {{TRACE}}..."
    echo "Config: config.smoke.yaml (2 seeds, 200 timesteps, max-steps=5)"
    snakemake \
        --configfile config.smoke.yaml \
        --config trace_name={{TRACE}} \
        --cores {{cpu_count}} 
    echo "✓ Smoke test complete. Outputs in result/{{TRACE}}/"

@run_full:
    echo "Running full production pipeline on {{TRACE}}..."
    echo "Config: config.yaml (5 seeds, 3M timesteps)"
    snakemake \
        --configfile config.yaml \
        --config trace_name={{TRACE}} \
        --cores {{cpu_count}}
    echo "✓ Full pipeline complete. Outputs in result/{{TRACE}}/"

@run_full_with_base:
    echo "Running full pipeline with baseline on {{TRACE}}..."
    snakemake \
        --configfile config.yaml \
        --config trace_name={{TRACE}} \
        result/{{TRACE}}/stats/stats_summary.json \
        result/{{TRACE}}/baseline/baseline_metadata.json \
        --cores {{cpu_count}}
    echo "✓ Full pipeline + baseline complete."

@run_baseline:
    echo "Running baseline scheduler (FCFS + best_fit) on {{TRACE}}..."
    snakemake \
        --configfile config.yaml \
        --config trace_name={{TRACE}} \
        result/{{TRACE}}/baseline/baseline_metadata.json \
        --cores {{cpu_count}}
    echo "✓ Baseline complete. Outputs in result/{{TRACE}}/baseline/"

# =============================================================================
# SLURM CLUSTER TARGETS
# =============================================================================

@dry_run_smoke_slurm:
    echo "Validating smoke DAG for cluster (no execution)..."
    snakemake --configfile config.smoke.yaml --profile profiles/slurm --dry-run --quiet

@dry_run_slurm:
    echo "Validating production DAG for cluster (no execution)..."
    snakemake --configfile config.yaml --profile profiles/slurm --dry-run --quiet

@run_smoke_slurm:
    echo "Submitting smoke test to SLURM on {{TRACE}}..."
    snakemake \
        --configfile config.smoke.yaml \
        --config trace_name={{TRACE}} \
        --profile profiles/slurm
    echo "✓ Smoke jobs submitted. Check squeue for status."

@run_full_slurm:
    echo "Submitting full pipeline to SLURM on {{TRACE}}..."
    snakemake \
        --configfile config.yaml \
        --config trace_name={{TRACE}} \
        --profile profiles/slurm
    echo "✓ Full pipeline submitted. Check squeue for status."

@run_full_with_base_slurm:
    echo "Submitting full pipeline + baseline to SLURM on {{TRACE}}..."
    snakemake \
        --configfile config.yaml \
        --config trace_name={{TRACE}} \
        result/{{TRACE}}/stats/stats_summary.json \
        result/{{TRACE}}/baseline/baseline_metadata.json \
        --profile profiles/slurm
    echo "✓ Submitted. Check squeue for status."

# Copy analysis outputs (and the winning algo's final models) off scratch into
# safe home storage. Idempotent — run it after eval to snapshot the
# aggregation/stats inputs early, and again at the end to grab stats + models.
@archive_results:
    ./src/archive_results.sh "{{TRACE}}" "{{ARCHIVE}}"

@slurm_report:
    echo "Generating SLURM efficiency report..."
    snakemake --configfile config.yaml --profile profiles/slurm --slurm-efficiency-report

@build_sif:
    echo "Building Apptainer .sif from Nix flake..."
    # 1. Build the container script using Nix
    nix build -L .#container -o nix-container-result 2>&1 | tee build.log
    
    # 2. Execute the script to stream the Docker archive to a tar file
    ./nix-container-result > DRL_env_docker.tar
    
    # 3. Build the Apptainer .sif directly from the Docker tarball
    apptainer build DRL_env.sif docker-archive://DRL_env_docker.tar
    
    # 4. Clean up the large temporary files
    rm -f DRL_env_docker.tar nix-container-result
    echo "✓ DRL_env.sif ready"



# =============================================================================
# DAG EXPORT TARGETS
# =============================================================================

@export_dag_detail:
    echo "Exporting job-level DAG for {{TRACE}}..."
    mkdir -p plots
    snakemake --configfile config.yaml --config trace_name={{TRACE}} --dag \
        | dot -Tsvg \
            -Grankdir=LR \
            -Gsplines=polyline \
            -Nshape=box \
            -Nstyle=rounded \
            -Efontsize=10 \
        -o plots/{{TRACE}}_dag_detail.svg
    echo "✓ Job DAG exported to plots/{{TRACE}}_dag_detail.svg"

@export_dag_overview:
    echo "Exporting rule-level DAG for {{TRACE}}..."
    mkdir -p plots
    snakemake --configfile config.yaml --config trace_name={{TRACE}} --rulegraph \
        | dot -Tsvg \
            -Grankdir=LR \
            -Gsplines=polyline \
            -Nshape=box \
            -Nstyle=rounded \
            -Efontsize=10 \
        -o plots/{{TRACE}}_dag_overview.svg
    echo "✓ Rule DAG exported to plots/{{TRACE}}_dag_overview.svg"

@export_dag:
    just export_dag_detail
    just export_dag_overview
    echo "✓ Both DAGs exported to plots/"
    echo "  - plots/{{TRACE}}_dag_detail.svg   (job-level; for appendix)"
    echo "  - plots/{{TRACE}}_dag_overview.svg (rule-level; for methodology)"

# =============================================================================
# MAINTENANCE TARGETS
# =============================================================================

@clean:
    echo "Cleaning pipeline outputs (data, code, and logs preserved)..."
    rm -rf result/ trained_model/ .snakemake/
    echo "✓ Clean complete"

@clean_all:
    echo "Cleaning all outputs including logs..."
    rm -rf result/ trained_model/ .snakemake/ logs/run_log.csv logs/baseline_run_log.csv logs/snakemake/
    echo "✓ Full clean complete"

@nix_develop:
    echo "Entering Nix develop environment..."
    nix develop -L

# =============================================================================
# NOTES
# =============================================================================
# Environment:      Nix (nix develop required before running)
# Snakemake:        9.4.3+
# just:             https://just.systems/man/en/
# TODO (future): Add Conda support as alternative to Nix
# =============================================================================
