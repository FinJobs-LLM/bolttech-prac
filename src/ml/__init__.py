"""Offline ML training/evaluation library for the claim-approval pipeline.

These modules are used by ``run_pipeline.py`` to train, optimize and evaluate
models. They are imported as ``ml.<module>`` (``src`` is on ``sys.path`` via the
entry points). The shared training⇄serving contract — ``config`` and
``model_factory`` (which defines the pickled ``ClaimModel``) — intentionally
stays at the top level of ``src`` so the saved model artifact and the FastAPI
serving apps keep resolving it unchanged.
"""
