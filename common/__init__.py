"""Support package for the vendored SAFE code under ``eval/safe``.

The SAFE files import ``common.modeling``, ``common.shared_config`` and
``common.utils`` from the original long-form-factuality repository. Those files
are not vendored here, so this package provides the same interfaces on top of
factscore's own LM backends, which lets ``eval/safe`` stay byte-identical.
"""
