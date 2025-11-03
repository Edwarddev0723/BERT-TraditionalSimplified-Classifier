from __future__ import annotations

import logging

from bert_ts_classifier.training import train as T
from bert_ts_classifier.utils.logging import setup_logger

MAX_EPOCHS = 5
MAX_LEN = 192


def test_parse_and_apply_overrides():
    base = {"train": {"max_epochs": 1, "lr": 1e-4}, "data": {"max_len": 128}}
    ov = T._parse_overrides(["train.max_epochs=5", "data.max_len=192", "model.use_transformers=false", "note=abc"]) 
    merged = T._apply_overrides(base, ov)  

    assert merged["train"]["max_epochs"] == MAX_EPOCHS
    assert merged["data"]["max_len"] == MAX_LEN
    assert merged["model"]["use_transformers"] is False
    assert merged["note"] == "abc"


def test_logger_setup_runs():
    # ensure logger can be set up without duplicate handlers
    lg = setup_logger("x", level=logging.DEBUG)
    lg.debug("ok")
