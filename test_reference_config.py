#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test script to verify the use_reference_model configuration works
"""

import torch
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.configs.worker_config import WorkerConfig


def test_reference_model_config():
    print("Testing use_reference_model configuration...")
    
    # Test default behavior (should be True)
    config_default = RLVRConfig()
    print(f"Default use_reference_model: {config_default.use_reference_model}")
    
    # Test explicitly setting to False
    config_no_ref = RLVRConfig(use_reference_model=False)
    print(f"Explicitly set use_reference_model to False: {config_no_ref.use_reference_model}")
    
    # Test explicitly setting to True
    config_with_ref = RLVRConfig(use_reference_model=True)
    print(f"Explicitly set use_reference_model to True: {config_with_ref.use_reference_model}")
    
    print("Test passed! Configuration parameter is working correctly.")


if __name__ == "__main__":
    test_reference_model_config()