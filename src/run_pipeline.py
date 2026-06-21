"""Backward-compatibility entry point — the implementation now lives in
``ml.run_pipeline``.

Kept so the documented command ``python src/run_pipeline.py`` keeps working
(``src`` is on ``sys.path``, so ``ml`` resolves). Equivalent to
``python -m ml.run_pipeline``.
"""
from ml.run_pipeline import main

if __name__ == "__main__":
    main()
