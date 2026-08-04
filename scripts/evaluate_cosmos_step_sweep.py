#!/usr/bin/env python3
"""Evaluate a trained TDV-QSM checkpoint on COSMOS at S=2, 10, 50, and 100."""

from qsm_tdv.evaluation.cosmos_step_sweep import main


if __name__ == "__main__":
    main()
