#!/usr/bin/env python3
"""
ColBERT to ONNX Converter
Converts ColBERT models to ONNX format compatible with transformers.js
"""

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
import argparse

import torch
import onnx
import onnxruntime as ort
from transformers import AutoConfig, AutoTokenizer
from pylate import models

from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxslim
from optimum.onnx.graph_transformations import check_and_save_model


@dataclass
class ConversionArguments:
    """Arguments for converting ColBERT models to ONNX."""

    model_id: str = field(
        metadata={"help": "Model identifier (e.g., mixedbread-ai/mxbai-edge-colbert-v0-17m)"}
    )

    output_parent_dir: str = field(
        default='./models/',
        metadata={"help": "Path where the converted model will be saved"}
    )

    quantize: bool = field(
        default=False,
        metadata={"help": "Whether to quantize the model"}
    )

    skip_onnxslim: bool = field(
        default=False,
        metadata={"help": "Whether to skip onnxslim optimization"}
    )

    device: str = field(
        default='cpu',
        metadata={"help": "Device to use for export (cpu/cuda)"}
    )

    opset_version: int = field(
        default=17,
        metadata={"help": "ONNX opset version"}
    )


class ColBERTONNXWrapper(torch.nn.Module):
    """ONNX-compatible wrapper for ColBERT models."""

    def __init__(self, modules: List[torch.nn.Module]):
        """
        Initialize wrapper with ColBERT modules.

        Args:
            modules: List of modules [transformer, dense1, dense2, ...]
        """
        super().__init__()
        self.modules_list = torch.nn.ModuleList(modules)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all ColBERT modules.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Final embeddings [batch, seq_len, output_dim]
        """
        # Start with transformer
        features = self.modules_list[0]({"input_ids": input_ids, "attention_mask": attention_mask})
        embeddings = features["token_embeddings"]

        # Apply remaining dense layers
        for module in self.modules_list[1:]:
            features = module({"token_embeddings": embeddings})
            embeddings = features["token_embeddings"]

        return embeddings


def detect_architecture(colbert_model) -> Tuple[List[torch.nn.Module], List[int]]:
    """
    Detect the architecture of a ColBERT model.

    Args:
        colbert_model: PyLate ColBERT model

    Returns:
        Tuple of (modules list, dimensions list)
    """
    print("\n" + "="*60)
    print("ARCHITECTURE DETECTION")
    print("="*60)

    # Extract modules
    modules = list(colbert_model.named_children())
    print(f"\nFound {len(modules)} modules:")

    module_list = []
    dimensions = []

    for idx, (name, module) in enumerate(modules):
        print(f"  [{idx}] {name}: {type(module).__name__}")
        module_list.append(module)

        # Try to detect output dimension
        if hasattr(module, 'out_features'):
            dim = module.out_features
            print(f"      → Output dimension: {dim}")
            dimensions.append(dim)
        elif hasattr(module, 'linear') and hasattr(module.linear, 'out_features'):
            dim = module.linear.out_features
            print(f"      → Output dimension: {dim}")
            dimensions.append(dim)

    # Test the model to get actual dimensions
    print("\nTesting model to verify dimensions...")
    dummy_input_ids = torch.randint(0, 50000, (1, 32), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 32), dtype=torch.long)

    with torch.no_grad():
        try:
            # Test transformer output
            transformer_out = module_list[0]({"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask})
            transformer_dim = transformer_out["token_embeddings"].shape[-1]
            if len(dimensions) == 0 or dimensions[0] != transformer_dim:
                dimensions.insert(0, transformer_dim)
                print(f"  ✓ Transformer output dimension: {transformer_dim}")

            # Test final output
            current_embeddings = transformer_out["token_embeddings"]
            for idx, module in enumerate(module_list[1:], 1):
                output = module({"token_embeddings": current_embeddings})
                current_embeddings = output["token_embeddings"]
                out_dim = current_embeddings.shape[-1]
                if idx >= len(dimensions) or dimensions[idx] != out_dim:
                    if idx < len(dimensions):
                        dimensions[idx] = out_dim
                    else:
                        dimensions.append(out_dim)
                    print(f"  ✓ Module {idx} output dimension: {out_dim}")

            final_dim = current_embeddings.shape[-1]
            print(f"  ✓ Final output dimension: {final_dim}")
        except Exception as e:
            print(f"  ⚠ Could not verify dimensions: {e}")

    print("\nArchitecture summary:")
    if len(dimensions) >= 2:
        arch_str = " → ".join([str(d) for d in dimensions])
        print(f"  {arch_str}")
    else:
        print(f"  {len(module_list)} modules detected")

    print("="*60 + "\n")

    return module_list, dimensions


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    opset_version: int = 17,
    device: str = "cpu",
    sequence_length: int = 128
) -> None:
    """
    Export model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
        sequence_length: Default sequence length for dummy input
    """
    print(f"\nExporting to ONNX: {output_path}")

    # Create dummy inputs
    export_device = torch.device(device)
    dummy_input_ids = torch.randint(0, 50000, (2, sequence_length), dtype=torch.long, device=export_device)
    dummy_attention_mask = torch.ones((2, sequence_length), dtype=torch.long, device=export_device)

    # Test model before export
    print("Testing model before export...")
    with torch.no_grad():
        test_output = model(dummy_input_ids, dummy_attention_mask)
        print(f"  Output shape: {test_output.shape}")

    # Export
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=opset_version,
        do_constant_folding=True
    )

    print("  ✓ Export complete")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    session = ort.InferenceSession(output_path)
    output_info = session.get_outputs()[0]
    print(f"  Output name: {output_info.name}")
    print(f"  Output shape: {output_info.shape}")
    print(f"  Output dimension: {output_info.shape[2]}")
    print("  ✓ Verification complete")


def quantize_model(input_path: str, output_path: str) -> None:
    """
    Quantize ONNX model to INT8.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save quantized model
    """
    print(f"\nQuantizing model...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    try:
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8,
        )
        print("  ✓ Quantization complete")

        # Compare sizes
        original_size = os.path.getsize(input_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Quantized size: {quantized_size:.2f} MB")
        print(f"  Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")

    except Exception as e:
        print(f"  ⚠ Quantization failed: {e}")
        raise


def copy_tokenizer_files(model_id: str, output_dir: str, tokenizer_dir: Optional[str] = None) -> None:
    """
    Copy tokenizer and config files to output directory.

    Args:
        model_id: Model identifier
        output_dir: Output directory
        tokenizer_dir: Optional directory containing tokenizer files
    """
    print("\nCopying tokenizer and config files...")

    # Try to load and save tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(output_dir)
        print("  ✓ Tokenizer files saved")
    except Exception as e:
        print(f"  ⚠ Could not save tokenizer: {e}")

    # Try to load and save config
    try:
        config = AutoConfig.from_pretrained(model_id)
        config.save_pretrained(output_dir)
        print("  ✓ Config file saved")
    except Exception as e:
        print(f"  ⚠ Could not save config: {e}")


def main():
    """Main conversion function."""

    parser = argparse.ArgumentParser(
        description="Convert ColBERT models to ONNX format"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model identifier (e.g., mixedbread-ai/mxbai-edge-colbert-v0-17m)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models/",
        help="Output parent directory (default: ./models/)"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model to INT8"
    )
    parser.add_argument(
        "--skip-onnxslim",
        action="store_true",
        help="Skip onnxslim optimization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for export"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ColBERT to ONNX Converter")
    print("="*60)
    print(f"\nModel ID: {args.model_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"Quantize: {args.quantize}")
    print(f"Device: {args.device}")

    # Create output directory
    output_model_folder = os.path.join(args.output_dir, args.model_id)
    os.makedirs(output_model_folder, exist_ok=True)
    os.makedirs(os.path.join(output_model_folder, 'onnx'), exist_ok=True)

    # Load ColBERT model with PyLate
    print("\n" + "="*60)
    print("LOADING MODEL")
    print("="*60)
    print(f"\nLoading ColBERT model with PyLate...")
    colbert_model = models.ColBERT(model_name_or_path=args.model_id, device=args.device)
    print("  ✓ Model loaded")

    # Detect architecture
    module_list, dimensions = detect_architecture(colbert_model)

    # Create ONNX wrapper
    print("\n" + "="*60)
    print("CREATING ONNX WRAPPER")
    print("="*60)
    onnx_model = ColBERTONNXWrapper(module_list)
    onnx_model.eval()
    onnx_model.to(args.device)

    # Test wrapper
    print("\nTesting ONNX wrapper...")
    dummy_input_ids = torch.randint(0, 50000, (2, 128), dtype=torch.long, device=args.device)
    dummy_attention_mask = torch.ones((2, 128), dtype=torch.long, device=args.device)

    with torch.no_grad():
        test_output = onnx_model(dummy_input_ids, dummy_attention_mask)
        print(f"  Output shape: {test_output.shape}")
        if len(dimensions) > 0 and test_output.shape[-1] == dimensions[-1]:
            print(f"  ✓ Correct! Model outputs {dimensions[-1]} dimensions")
        else:
            print(f"  ⚠ Unexpected output dimension: {test_output.shape[-1]}")

    # Export to ONNX
    print("\n" + "="*60)
    print("EXPORTING TO ONNX")
    print("="*60)
    model_path = os.path.join(output_model_folder, "model.onnx")
    export_to_onnx(
        onnx_model,
        model_path,
        opset_version=args.opset_version,
        device=args.device
    )

    # Apply onnxslim
    if not args.skip_onnxslim:
        print("\n" + "="*60)
        print("OPTIMIZING WITH ONNXSLIM")
        print("="*60)
        try:
            print("Applying onnxslim optimization...")
            slimmed_model = onnxslim.slim(model_path)
            check_and_save_model(slimmed_model, model_path)
            print("  ✓ Optimization complete")
        except Exception as e:
            print(f"  ⚠ Optimization failed: {e}")

    # Quantize if requested
    if args.quantize:
        print("\n" + "="*60)
        print("QUANTIZATION")
        print("="*60)
        quantized_path = os.path.join(output_model_folder, "model_quantized.onnx")
        quantize_model(model_path, quantized_path)

    # Copy tokenizer and config files
    copy_tokenizer_files(args.model_id, output_model_folder)

    # Move .onnx files to onnx subfolder
    print("\nOrganizing output files...")
    for file in os.listdir(output_model_folder):
        if file.endswith(('.onnx', '.onnx_data')):
            src = os.path.join(output_model_folder, file)
            dst = os.path.join(output_model_folder, 'onnx', file)
            shutil.move(src, dst)
            print(f"  Moved {file} to onnx/")

    # Save metadata
    metadata = {
        "model_id": args.model_id,
        "architecture": {
            "modules": len(module_list),
            "dimensions": dimensions,
        },
        "quantized": args.quantize,
        "opset_version": args.opset_version,
    }

    with open(os.path.join(output_model_folder, 'conversion_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create README
    arch_str = " → ".join([str(d) for d in dimensions]) if dimensions else f"{len(module_list)} modules"
    readme_content = f"""# {args.model_id.split('/')[-1]} (ONNX)

This is a ColBERT model converted to ONNX format.

## Architecture
{arch_str}

## Files
- `onnx/model.onnx` - Main ONNX model
{"- `onnx/model_quantized.onnx` - Quantized INT8 model" if args.quantize else ""}
- `tokenizer.json` - Tokenizer configuration
- `config.json` - Model configuration

## Usage
Use with transformers.js or any ONNX runtime for ColBERT-style retrieval.

## Conversion Info
- Model ID: {args.model_id}
- Opset version: {args.opset_version}
- Quantized: {args.quantize}
"""

    with open(os.path.join(output_model_folder, 'README.md'), 'w') as f:
        f.write(readme_content)

    # Final summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {output_model_folder}")
    print(f"Architecture: {arch_str}")
    print(f"Quantized: {args.quantize}")
    print("\nFiles created:")
    for root, dirs, files in os.walk(output_model_folder):
        level = root.replace(output_model_folder, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            size = os.path.getsize(os.path.join(root, file)) / (1024 * 1024)
            print(f"{subindent}{file} ({size:.2f} MB)")

    print("\n" + "="*60)
    print("✓ Ready to use with transformers.js!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
