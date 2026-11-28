#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thin CLI wrapper around ``whitematteranalysis.pipelines.org``."""

import contextlib
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parents[1]
if (_repo_root / 'whitematteranalysis').is_dir():
    _repo_str = str(_repo_root)
    if _repo_str not in sys.path:
        sys.path.insert(0, _repo_str)

from whitematteranalysis.pipelines import org as _org
from whitematteranalysis.pipelines.org import (
    BatchResult,
    COMMISSURAL_TRACTS,
    HEMISPHERIC_TRACTS,
    ORGConfig,
    ORGParcellationPipeline,
    SubjectResult,
    _AtlasSession,
    _PRESETS,
    _assemble_single_tract,
    _build_arg_parser,
    _compute_cluster_outlier_scores,
    _default_lz4_compression,
    _discover_inputs,
    _empty_like_schema,
    _invoke_slicer_hardening,
    parcellate_batch,
    parcellate_subject,
)

_SYNCED_HOOKS = ('_assemble_single_tract', '_invoke_slicer_hardening')


@contextlib.contextmanager
def _sync_monkeypatched_hooks():
    """Forward module-level monkeypatches on this CLI script into pipelines.org."""
    module_globals = globals()
    saved = {}
    for name in _SYNCED_HOOKS:
        current = module_globals.get(name)
        original = getattr(_org, name)
        if current is not original:
            saved[name] = original
            setattr(_org, name, current)
    try:
        yield
    finally:
        for name, original in saved.items():
            setattr(_org, name, original)


def run_pipeline(args):
    with _sync_monkeypatched_hooks():
        return _org.run_pipeline(args)


def main(argv=None):
    with _sync_monkeypatched_hooks():
        return _org.main(argv)


if __name__ == '__main__':
    main()
