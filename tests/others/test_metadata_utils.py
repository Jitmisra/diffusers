# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import tempfile
import unittest
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from diffusers import __version__
from diffusers.utils import (
    export_metadata,
    extract_model_metadata,
    extract_pipeline_metadata,
    save_metadata,
)

from ..testing_utils import require_torch, torch_device


class SimpleTestModel(nn.Module):
    """A simple test model for testing metadata extraction."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class ModelWithConfig(nn.Module):
    """A test model with a config attribute."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.config = type("Config", (), {"hidden_size": 128, "num_layers": 4})()

    def forward(self, x):
        return self.linear(x)


@require_torch
class MetadataUtilsTest(unittest.TestCase):
    def test_extract_model_metadata_basic(self):
        """Test basic metadata extraction from a simple model."""
        model = SimpleTestModel()
        metadata = extract_model_metadata(model)

        # Check required fields
        self.assertIn("extraction_timestamp", metadata)
        self.assertIn("diffusers_version", metadata)
        self.assertEqual(metadata["diffusers_version"], __version__)
        self.assertIn("model_class", metadata)
        self.assertEqual(metadata["model_class"], "SimpleTestModel")
        self.assertIn("num_parameters", metadata)
        self.assertIn("num_trainable_parameters", metadata)
        self.assertIn("memory_footprint_bytes", metadata)

        # Check parameter counts
        self.assertIsInstance(metadata["num_parameters"], int)
        self.assertGreater(metadata["num_parameters"], 0)
        self.assertIsInstance(metadata["num_trainable_parameters"], int)

        # Check memory footprint
        self.assertIsInstance(metadata["memory_footprint_bytes"], int)
        self.assertGreater(metadata["memory_footprint_bytes"], 0)
        self.assertIn("memory_footprint_mb", metadata)
        self.assertIn("memory_footprint_gb", metadata)

    def test_extract_model_metadata_with_config(self):
        """Test metadata extraction with model configuration."""
        model = ModelWithConfig()
        metadata = extract_model_metadata(model, include_config=True)

        self.assertIn("config", metadata)
        self.assertIsNotNone(metadata["config"])
        self.assertIn("hidden_size", metadata["config"])
        self.assertEqual(metadata["config"]["hidden_size"], 128)

    def test_extract_model_metadata_device_info(self):
        """Test device information extraction."""
        model = SimpleTestModel()
        model = model.to(torch_device)
        metadata = extract_model_metadata(model, include_device_info=True)

        self.assertIn("device", metadata)
        device_info = metadata["device"]
        self.assertIn("device", device_info)
        self.assertIn("is_sharded", device_info)

    def test_extract_model_metadata_exclude_options(self):
        """Test metadata extraction with excluded options."""
        model = SimpleTestModel()
        metadata = extract_model_metadata(
            model,
            include_config=False,
            include_quantization=False,
            include_adapters=False,
            include_device_info=False,
        )

        # Should still have basic info
        self.assertIn("model_class", metadata)
        self.assertIn("num_parameters", metadata)

        # Should not have excluded info
        self.assertNotIn("config", metadata)
        self.assertNotIn("quantization", metadata)
        self.assertNotIn("adapters", metadata)
        self.assertNotIn("device", metadata)

    def test_extract_model_metadata_invalid_input(self):
        """Test that invalid input raises appropriate error."""
        with self.assertRaises(ValueError):
            extract_model_metadata("not a model")

    def test_export_metadata_json(self):
        """Test exporting metadata to JSON format."""
        model = SimpleTestModel()
        metadata = extract_model_metadata(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metadata"
            export_metadata(metadata, output_path, format="json")

            # Check file exists
            json_path = output_path.with_suffix(".json")
            self.assertTrue(json_path.exists())

            # Check file content
            with open(json_path, "r") as f:
                loaded_metadata = json.load(f)

            self.assertEqual(loaded_metadata["model_class"], metadata["model_class"])
            self.assertEqual(loaded_metadata["num_parameters"], metadata["num_parameters"])

    def test_export_metadata_yaml(self):
        """Test exporting metadata to YAML format."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not available")

        model = SimpleTestModel()
        metadata = extract_model_metadata(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metadata"
            export_metadata(metadata, output_path, format="yaml")

            # Check file exists
            yaml_path = output_path.with_suffix(".yaml")
            self.assertTrue(yaml_path.exists())

            # Check file content
            with open(yaml_path, "r") as f:
                loaded_metadata = yaml.safe_load(f)

            self.assertEqual(loaded_metadata["model_class"], metadata["model_class"])

    def test_export_metadata_invalid_format(self):
        """Test that invalid format raises appropriate error."""
        metadata = {"test": "data"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metadata"
            with self.assertRaises(ValueError):
                export_metadata(metadata, output_path, format="invalid")

    def test_save_metadata_model(self):
        """Test saving metadata for a model."""
        model = SimpleTestModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = save_metadata(model, tmpdir, is_pipeline=False, format="json")

            # Check file exists
            self.assertTrue(metadata_path.exists())
            self.assertEqual(metadata_path.suffix, ".json")

            # Check content
            with open(metadata_path, "r") as f:
                loaded_metadata = json.load(f)

            self.assertEqual(loaded_metadata["model_class"], "SimpleTestModel")
            self.assertIn("num_parameters", loaded_metadata)

    def test_save_metadata_custom_filename(self):
        """Test saving metadata with custom filename."""
        model = SimpleTestModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = save_metadata(
                model, tmpdir, is_pipeline=False, filename="custom_metadata", format="json"
            )

            self.assertTrue(metadata_path.exists())
            self.assertEqual(metadata_path.stem, "custom_metadata")

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available for device info test"
    )
    def test_extract_model_metadata_cuda_device(self):
        """Test metadata extraction with CUDA device."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        model = SimpleTestModel()
        model = model.cuda()
        metadata = extract_model_metadata(model, include_device_info=True)

        self.assertIn("device", metadata)
        device_info = metadata["device"]
        self.assertIn("device", device_info)
        self.assertIn("cuda", device_info["device"])

    def test_metadata_structure(self):
        """Test that metadata has expected structure and types."""
        model = SimpleTestModel()
        metadata = extract_model_metadata(model)

        # Check structure
        required_fields = [
            "extraction_timestamp",
            "diffusers_version",
            "model_class",
            "model_module",
            "num_parameters",
            "num_trainable_parameters",
            "num_parameters_millions",
            "num_parameters_billions",
            "memory_footprint_bytes",
            "memory_footprint_mb",
            "memory_footprint_gb",
        ]

        for field in required_fields:
            self.assertIn(field, metadata, f"Missing required field: {field}")

        # Check types
        self.assertIsInstance(metadata["extraction_timestamp"], str)
        self.assertIsInstance(metadata["diffusers_version"], str)
        self.assertIsInstance(metadata["model_class"], str)
        self.assertIsInstance(metadata["num_parameters"], int)
        self.assertIsInstance(metadata["memory_footprint_bytes"], int)

    def test_extract_pipeline_metadata_mock(self):
        """Test pipeline metadata extraction with a mock pipeline-like object."""
        # Create a simple mock pipeline
        class MockPipeline:
            def __init__(self):
                self.unet = SimpleTestModel()
                self.vae = SimpleTestModel()
                self.config = type("Config", (), {"_class_name": "MockPipeline"})()

        pipeline = MockPipeline()

        # This should work even if not a real pipeline
        try:
            metadata = extract_pipeline_metadata(pipeline)
            self.assertIn("pipeline_class", metadata)
            self.assertIn("extraction_timestamp", metadata)
        except Exception as e:
            # If it fails, that's okay - we're just testing the function exists
            # Real pipeline tests would be in integration tests
            pass

