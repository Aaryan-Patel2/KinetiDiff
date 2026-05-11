"""Verify the 3 kinetidiff console-script entry points resolve to callable main() functions."""
import importlib


def test_generate_main_is_callable():
    mod = importlib.import_module("kinetidiff.generation.generate_with_vina_guidance")
    assert callable(getattr(mod, "main", None)), (
        "kinetidiff.generation.generate_with_vina_guidance must expose a callable main()"
    )


def test_generate_multi_main_is_callable():
    mod = importlib.import_module("kinetidiff.generation.generate_multi_objective")
    assert callable(getattr(mod, "main", None)), (
        "kinetidiff.generation.generate_multi_objective must expose a callable main()"
    )


def test_train_main_is_callable():
    mod = importlib.import_module("kinetidiff.train.train_affinity")
    assert callable(getattr(mod, "main", None)), (
        "kinetidiff.train.train_affinity must expose a callable main()"
    )
