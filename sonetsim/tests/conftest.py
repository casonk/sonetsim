"""Shared pytest fixtures for sonetsim tests."""

from importlib.util import find_spec

pytest_plugins = ["dyno_lab.fixtures"] if find_spec("dyno_lab") else []
