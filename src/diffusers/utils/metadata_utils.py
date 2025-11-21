# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""
Utilities for extracting and exporting model metadata.

This module provides functions to extract comprehensive metadata from diffusion models
and pipelines, including architecture information, parameter counts, memory footprint,
quantization details, adapters, and more. The metadata can be exported in various formats
for integration with model catalogs and sharing platforms.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .. import __version__
from .import_utils import (
    is_bitsandbytes_available,
    is_peft_available,
    is_torch_available,
    is_torchao_available,
)
from .logging import get_logger

logger = get_logger(__name__)

if is_torch_available():
    import torch
    import torch.nn as nn


def extract_model_metadata(
    model: Union[nn.Module, Any],
    include_config: bool = True,
    include_quantization: bool = True,
    include_adapters: bool = True,
    include_device_info: bool = True,
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a diffusion model.

    This function extracts detailed information about a model including:
    - Architecture and class information
    - Parameter counts and memory footprint
    - Quantization information (if applicable)
    - Adapter information (LoRA, etc.)
    - Device placement information
    - Model configuration (if available)

    Args:
        model (`torch.nn.Module` or compatible):
            The model to extract metadata from. Should be a PyTorch model or a diffusers model.
        include_config (`bool`, *optional*, defaults to `True`):
            Whether to include model configuration in the metadata.
        include_quantization (`bool`, *optional*, defaults to `True`):
            Whether to extract quantization information.
        include_adapters (`bool`, *optional*, defaults to `True`):
            Whether to extract adapter (LoRA, etc.) information.
        include_device_info (`bool`, *optional*, defaults to `True`):
            Whether to include device placement information.

    Returns:
        `Dict[str, Any]`: A dictionary containing comprehensive model metadata.

    Example:
        ```python
        from diffusers import UNet2DConditionModel
        from diffusers.utils import extract_model_metadata

        model = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet")
        metadata = extract_model_metadata(model)
        print(metadata["num_parameters"])
        ```
    """
    if not is_torch_available():
        raise ImportError("PyTorch is required to extract model metadata.")

    if not isinstance(model, nn.Module):
        raise ValueError(f"Model must be a torch.nn.Module, got {type(model)}")

    metadata: Dict[str, Any] = {
        "extraction_timestamp": datetime.now(UTC).isoformat(),
        "diffusers_version": __version__,
    }

    # Basic model information
    metadata["model_class"] = model.__class__.__name__
    metadata["model_module"] = model.__class__.__module__

    # Parameter information
    try:
        if hasattr(model, "num_parameters"):
            metadata["num_parameters"] = model.num_parameters(only_trainable=False)
            metadata["num_trainable_parameters"] = model.num_parameters(only_trainable=True)
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            metadata["num_parameters"] = total_params
            metadata["num_trainable_parameters"] = trainable_params

        metadata["num_parameters_millions"] = round(metadata["num_parameters"] / 1e6, 2)
        metadata["num_parameters_billions"] = round(metadata["num_parameters"] / 1e9, 2)
    except Exception as e:
        logger.warning(f"Could not extract parameter information: {e}")
        metadata["num_parameters"] = None

    # Memory footprint
    try:
        if hasattr(model, "get_memory_footprint"):
            memory_bytes = model.get_memory_footprint(return_buffers=True)
        else:
            # Fallback calculation
            memory_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            memory_bytes += sum(buf.numel() * buf.element_size() for buf in model.buffers())

        metadata["memory_footprint_bytes"] = memory_bytes
        metadata["memory_footprint_mb"] = round(memory_bytes / (1024**2), 2)
        metadata["memory_footprint_gb"] = round(memory_bytes / (1024**3), 2)
    except Exception as e:
        logger.warning(f"Could not extract memory footprint: {e}")
        metadata["memory_footprint_bytes"] = None

    # Model configuration
    if include_config:
        try:
            if hasattr(model, "config"):
                config = model.config
                # Convert config to dict if it's a dataclass or similar
                if hasattr(config, "to_dict"):
                    metadata["config"] = config.to_dict()
                elif hasattr(config, "__dict__"):
                    # Filter out non-serializable items
                    config_dict = {}
                    for key, value in config.__dict__.items():
                        if not key.startswith("_"):
                            try:
                                json.dumps(value)  # Test if serializable
                                config_dict[key] = value
                            except (TypeError, ValueError):
                                config_dict[key] = str(value)
                    if config_dict:
                        metadata["config"] = config_dict
                    else:
                        # Fall through to try dir() approach
                        pass
                if "config" not in metadata:
                    # For simple objects with attributes (like our test model)
                    # Try to get attributes directly
                    config_dict = {}
                    for attr in dir(config):
                        if not attr.startswith("_") and not callable(getattr(config, attr, None)):
                            try:
                                value = getattr(config, attr)
                                # Skip methods and special attributes
                                if not callable(value):
                                    json.dumps(value)
                                    config_dict[attr] = value
                            except (TypeError, ValueError, AttributeError):
                                pass
                    if config_dict:
                        metadata["config"] = config_dict
                    else:
                        metadata["config"] = str(config)
                else:
                    metadata["config"] = str(config)
        except Exception as e:
            logger.warning(f"Could not extract configuration: {e}")
            metadata["config"] = None

    # Quantization information
    if include_quantization:
        quantization_info = _extract_quantization_info(model)
        if quantization_info:
            metadata["quantization"] = quantization_info

    # Adapter information (LoRA, etc.)
    if include_adapters:
        adapter_info = _extract_adapter_info(model)
        if adapter_info:
            metadata["adapters"] = adapter_info

    # Device information
    if include_device_info:
        device_info = _extract_device_info(model)
        if device_info:
            metadata["device"] = device_info

    # Additional model-specific information
    try:
        # Check for IP-Adapter
        if hasattr(model, "attn_processors"):
            processors = model.attn_processors
            processor_types = [v.__class__.__name__ for v in processors.values()]
            if any("IPAdapter" in ptype for ptype in processor_types):
                metadata["has_ip_adapter"] = True
                # Extract IP-Adapter scales if available
                scales = {
                    k: v.scale
                    for k, v in processors.items()
                    if hasattr(v, "scale") and "IPAdapter" in v.__class__.__name__
                }
                if scales:
                    metadata["ip_adapter_scales"] = scales
    except Exception:
        pass

    # Check for hooks (e.g., from accelerate)
    try:
        if hasattr(model, "_hf_hook"):
            metadata["has_accelerate_hook"] = True
            if hasattr(model._hf_hook, "execution_device"):
                metadata["execution_device"] = str(model._hf_hook.execution_device)
    except Exception:
        pass

    return metadata


def extract_pipeline_metadata(
    pipeline: Any,
    include_components: bool = True,
    include_config: bool = True,
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a diffusion pipeline.

    This function extracts metadata from all components of a pipeline including
    models, schedulers, and processors.

    Args:
        pipeline:
            The diffusion pipeline to extract metadata from.
        include_components (`bool`, *optional*, defaults to `True`):
            Whether to extract metadata from individual pipeline components.
        include_config (`bool`, *optional*, defaults to `True`):
            Whether to include pipeline configuration.

    Returns:
        `Dict[str, Any]`: A dictionary containing comprehensive pipeline metadata.

    Example:
        ```python
        from diffusers import DiffusionPipeline
        from diffusers.utils import extract_pipeline_metadata

        pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        metadata = extract_pipeline_metadata(pipeline)
        ```
    """
    metadata: Dict[str, Any] = {
        "extraction_timestamp": datetime.now(UTC).isoformat(),
        "diffusers_version": __version__,
        "pipeline_class": pipeline.__class__.__name__,
        "pipeline_module": pipeline.__class__.__module__,
    }

    # Pipeline configuration
    if include_config:
        try:
            if hasattr(pipeline, "config"):
                config = pipeline.config
                if hasattr(config, "to_dict"):
                    metadata["pipeline_config"] = config.to_dict()
                elif hasattr(config, "__dict__"):
                    config_dict = {}
                    for key, value in config.__dict__.items():
                        if not key.startswith("_"):
                            try:
                                json.dumps(value)
                                config_dict[key] = value
                            except (TypeError, ValueError):
                                config_dict[key] = str(value)
                    metadata["pipeline_config"] = config_dict
        except Exception as e:
            logger.warning(f"Could not extract pipeline configuration: {e}")

    # Component metadata
    if include_components:
        components_metadata = {}
        try:
            # Get all registered components
            if hasattr(pipeline, "config") and hasattr(pipeline.config, "__dict__"):
                for component_name in pipeline.config.__dict__.keys():
                    if not component_name.startswith("_") and hasattr(pipeline, component_name):
                        component = getattr(pipeline, component_name)
                        if component is not None:
                            try:
                                if is_torch_available() and isinstance(component, nn.Module):
                                    components_metadata[component_name] = extract_model_metadata(component)
                                else:
                                    # For non-model components (schedulers, processors, etc.)
                                    components_metadata[component_name] = {
                                        "component_class": component.__class__.__name__,
                                        "component_module": component.__class__.__module__,
                                    }
                            except Exception as e:
                                logger.warning(f"Could not extract metadata for component {component_name}: {e}")
                                components_metadata[component_name] = {
                                    "component_class": component.__class__.__name__,
                                    "error": str(e),
                                }

            metadata["components"] = components_metadata
        except Exception as e:
            logger.warning(f"Could not extract component metadata: {e}")

    # Calculate total pipeline statistics
    try:
        total_params = 0
        total_memory = 0
        if "components" in metadata:
            for comp_name, comp_meta in metadata["components"].items():
                if isinstance(comp_meta, dict):
                    if "num_parameters" in comp_meta and comp_meta["num_parameters"] is not None:
                        total_params += comp_meta["num_parameters"]
                    if "memory_footprint_bytes" in comp_meta and comp_meta["memory_footprint_bytes"] is not None:
                        total_memory += comp_meta["memory_footprint_bytes"]

        if total_params > 0:
            metadata["total_parameters"] = total_params
            metadata["total_parameters_millions"] = round(total_params / 1e6, 2)
            metadata["total_parameters_billions"] = round(total_params / 1e9, 2)

        if total_memory > 0:
            metadata["total_memory_footprint_bytes"] = total_memory
            metadata["total_memory_footprint_gb"] = round(total_memory / (1024**3), 2)
    except Exception as e:
        logger.warning(f"Could not calculate total pipeline statistics: {e}")

    return metadata


def _extract_quantization_info(model: nn.Module) -> Optional[Dict[str, Any]]:
    """Extract quantization information from a model."""
    quantization_info = {}

    # Check for bitsandbytes quantization
    if is_bitsandbytes_available():
        try:
            import bitsandbytes as bnb

            has_4bit = False
            has_8bit = False
            for module in model.modules():
                if isinstance(module, bnb.nn.Linear4bit):
                    has_4bit = True
                elif isinstance(module, bnb.nn.Linear8bitLt):
                    has_8bit = True

            if has_4bit:
                quantization_info["bitsandbytes_4bit"] = True
            if has_8bit:
                quantization_info["bitsandbytes_8bit"] = True
        except Exception:
            pass

    # Check for torchao quantization
    if is_torchao_available():
        try:
            # Check for quantized tensors
            for name, param in model.named_parameters():
                param_type = type(param).__name__
                if "Quantized" in param_type or "quantized" in param_type.lower():
                    quantization_info["torchao"] = True
                    quantization_info["quantized_layers"] = quantization_info.get("quantized_layers", [])
                    quantization_info["quantized_layers"].append(name)
                    break
        except Exception:
            pass

    # Check for hf_quantizer attribute (diffusers quantization)
    if hasattr(model, "hf_quantizer"):
        quantization_info["diffusers_quantizer"] = True
        quantizer = model.hf_quantizer
        if hasattr(quantizer, "quantization_config"):
            config = quantizer.quantization_config
            if hasattr(config, "__dict__"):
                quant_config_dict = {}
                for key, value in config.__dict__.items():
                    if not key.startswith("_"):
                        try:
                            json.dumps(value)
                            quant_config_dict[key] = value
                        except (TypeError, ValueError):
                            quant_config_dict[key] = str(value)
                quantization_info["quantization_config"] = quant_config_dict

    return quantization_info if quantization_info else None


def _extract_adapter_info(model: nn.Module) -> Optional[Dict[str, Any]]:
    """Extract adapter (LoRA, etc.) information from a model."""
    if not is_peft_available():
        return None

    adapter_info = {}

    try:
        from peft.tuners.tuners_utils import BaseTunerLayer

        adapter_names = set()
        adapter_types = set()

        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "active_adapters"):
                    adapter_names.update(module.active_adapters)
                elif hasattr(module, "active_adapter"):
                    adapter_names.add(module.active_adapter)

                adapter_types.add(module.__class__.__name__)

        if adapter_names:
            adapter_info["adapter_names"] = list(adapter_names)
        if adapter_types:
            adapter_info["adapter_types"] = list(adapter_types)

        # Check for peft_config
        if hasattr(model, "peft_config"):
            peft_configs = {}
            for adapter_name, config in model.peft_config.items():
                if hasattr(config, "to_dict"):
                    peft_configs[adapter_name] = config.to_dict()
                else:
                    peft_configs[adapter_name] = str(config)
            adapter_info["peft_configs"] = peft_configs

    except Exception as e:
        logger.warning(f"Could not extract adapter information: {e}")
        return None

    return adapter_info if adapter_info else None


def _extract_device_info(model: nn.Module) -> Optional[Dict[str, Any]]:
    """Extract device placement information from a model."""
    device_info = {}

    try:
        # Get device of first parameter
        first_param = next(model.parameters(), None)
        if first_param is not None:
            device_info["device"] = str(first_param.device)

        # Check for multiple devices (model sharding)
        devices = set()
        for param in model.parameters():
            devices.add(str(param.device))
        if len(devices) > 1:
            device_info["devices"] = list(devices)
            device_info["is_sharded"] = True
        else:
            device_info["is_sharded"] = False

        # Check for accelerate hook
        if hasattr(model, "_hf_hook"):
            hook = model._hf_hook
            if hasattr(hook, "execution_device"):
                device_info["execution_device"] = str(hook.execution_device)
            if hasattr(hook, "offload"):
                device_info["offload_enabled"] = hook.offload

    except Exception as e:
        logger.warning(f"Could not extract device information: {e}")
        return None

    return device_info if device_info else None


def export_metadata(
    metadata: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "json",
    indent: int = 2,
) -> None:
    """
    Export metadata to a file in the specified format.

    Args:
        metadata (`Dict[str, Any]`):
            The metadata dictionary to export.
        output_path (`str` or `Path`):
            Path where the metadata file should be saved.
        format (`str`, *optional*, defaults to `"json"`):
            Export format. Supported formats: "json", "yaml".
        indent (`int`, *optional*, defaults to `2`):
            Indentation level for JSON/YAML output.

    Example:
        ```python
        from diffusers.utils import extract_model_metadata, export_metadata

        metadata = extract_model_metadata(model)
        export_metadata(metadata, "model_metadata.json")
        ```
    """
    output_path = Path(output_path)

    if format.lower() == "json":
        output_path = output_path.with_suffix(".json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=indent, ensure_ascii=False)
        logger.info(f"Metadata exported to {output_path}")

    elif format.lower() == "yaml":
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML export. Install it with `pip install pyyaml`")

        output_path = output_path.with_suffix(".yaml")
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False, indent=indent, allow_unicode=True)
        logger.info(f"Metadata exported to {output_path}")

    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats: 'json', 'yaml'")


def save_metadata(
    model_or_pipeline: Union[nn.Module, Any],
    save_directory: Union[str, Path],
    is_pipeline: bool = False,
    filename: Optional[str] = None,
    format: str = "json",
) -> Path:
    """
    Extract and save metadata from a model or pipeline to a directory.

    This is a convenience function that combines extraction and export.

    Args:
        model_or_pipeline:
            The model or pipeline to extract metadata from.
        save_directory (`str` or `Path`):
            Directory where the metadata file should be saved.
        is_pipeline (`bool`, *optional*, defaults to `False`):
            Whether the input is a pipeline (True) or a model (False).
        filename (`str`, *optional*):
            Name of the metadata file (without extension). If None, defaults to "model_metadata" or "pipeline_metadata".
        format (`str`, *optional*, defaults to `"json"`):
            Export format. Supported formats: "json", "yaml".

    Returns:
        `Path`: Path to the saved metadata file.

    Example:
        ```python
        from diffusers import UNet2DConditionModel
        from diffusers.utils import save_metadata

        model = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet")
        metadata_path = save_metadata(model, "./my_model", format="json")
        ```
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    # Extract metadata
    if is_pipeline:
        metadata = extract_pipeline_metadata(model_or_pipeline)
        default_filename = "pipeline_metadata"
    else:
        metadata = extract_model_metadata(model_or_pipeline)
        default_filename = "model_metadata"

    filename = filename or default_filename
    output_path = save_directory / filename

    # Export metadata
    export_metadata(metadata, output_path, format=format)

    return output_path.with_suffix(f".{format.lower()}")

