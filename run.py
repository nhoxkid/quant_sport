"""
One-click runner for the entire NFL Totals ODE project (Phases 1-4).
Double-click this file or run: python run.py

All outputs are saved to: C:/Code/ODE_coupled/data_file/
  data_file/
    phase1/    — pipeline CSV + validation report (copied from pipeline/data/processed/)
    phase2/    — 10 statistical modeling figures
    phase3/    — 9 ODE model figures
    phase4/    — 9 simulation figures

The pipeline (Phase 1) uses cached data after the first run,
so subsequent runs are much faster.
"""
import sys
import os
import time
import shutil
import logging
from pathlib import Path

# Ensure the project root is on the path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

DATA_DIR = ROOT / "data_file"


def run_phase1():
    """Phase 1: Data pipeline — collect, merge, validate."""
    print("\n" + "=" * 60)
    print("  PHASE 1: DATA PIPELINE")
    print("=" * 60)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    from pipeline.run_pipeline import run
    exit_code = run()

    # Copy outputs to data_file/phase1/
    p1_dir = DATA_DIR / "phase1"
    p1_dir.mkdir(parents=True, exist_ok=True)
    src_dir = ROOT / "pipeline" / "data" / "processed"
    for f in src_dir.glob("*"):
        if f.is_file():
            shutil.copy2(f, p1_dir / f.name)

    if exit_code == 0:
        print("  Phase 1 PASSED. Data saved to data_file/phase1/")
    else:
        print("  Phase 1 FAILED validation. Check report.")
    return exit_code


def run_phase2():
    """Phase 2: Statistical modeling — regressions, efficiency tests, figures."""
    print("\n" + "=" * 60)
    print("  PHASE 2: STATISTICAL MODELING")
    print("=" * 60)

    from phase2_statistical_modeling import main as p2_main
    p2_main()
    print("  Phase 2 complete. Figures saved to data_file/phase2/")


def run_phase3():
    """Phase 3: ODE model — calibration, synthetic validation, simulation."""
    print("\n" + "=" * 60)
    print("  PHASE 3: ODE MODEL")
    print("=" * 60)

    from phase3_ode_model import main as p3_main
    p3_main()
    print("  Phase 3 complete. Figures saved to data_file/phase3/")


def run_phase4():
    """Phase 4: Simulation — SDE Monte Carlo, fan charts, closing errors."""
    print("\n" + "=" * 60)
    print("  PHASE 4: SIMULATION MODULE")
    print("=" * 60)

    from phase4_simulation import main as p4_main
    p4_main()
    print("  Phase 4 complete. Figures saved to data_file/phase4/")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  NFL TOTALS ODE PROJECT — FULL RUN")
    print("  All outputs -> C:/Code/ODE_coupled/data_file/")
    print("=" * 60)

    start = time.time()

    # Phase 1: Data pipeline
    p1_code = run_phase1()
    if p1_code != 0:
        print("\nPhase 1 failed. Fix data issues before running Phases 2-4.")
        input("\nPress Enter to exit...")
        sys.exit(1)

    # Phase 2: Statistical modeling
    run_phase2()

    # Phase 3: ODE model
    run_phase3()

    # Phase 4: Simulation
    run_phase4()

    elapsed = time.time() - start

    # Final summary
    print("\n" + "=" * 60)
    print("  ALL PHASES COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.0f} seconds")
    print()
    print("  Output directory: C:/Code/ODE_coupled/data_file/")
    print()

    # Count files per phase
    for phase in ["phase1", "phase2", "phase3", "phase4"]:
        d = DATA_DIR / phase
        if d.exists():
            n = len(list(d.glob("*")))
            print(f"    {phase}/  — {n} files")

    print()
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
