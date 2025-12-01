# ColBERT ONNX Converter

Convert ColBERT models to ONNX format for use with transformers.js. Validated with multiple models showing 2x speed improvements and 75% size reductions.

## Quick Start

```bash
# Install dependencies
uv sync

# Convert a model
uv run convert --model-id mixedbread-ai/mxbai-edge-colbert-v0-17m --quantize

# Validate the conversion
uv run validate --model-id mixedbread-ai/mxbai-edge-colbert-v0-17m
```

## Features

- ✅ **Proven Results**: 2x faster inference, 75% smaller models
- ✅ **Perfect Accuracy**: Full-precision ONNX matches PyTorch exactly
- ✅ **Automatic Detection**: Detects ColBERT architecture automatically
- ✅ **Quantization Support**: INT8 quantization with minimal accuracy loss
- ✅ **Multiple Models Tested**: Works with mixedbread, lightonai, and more

## Validation Results

We've thoroughly tested the converter with multiple ColBERT models:

### mixedbread-ai/mxbai-edge-colbert-v0-17m
- **Embeddings**: 100% match (cosine similarity 1.0)
- **Speed**: 2.11x faster with quantization
- **Size**: 65 MB → 16 MB (75% reduction)

### lightonai/Reason-ModernColBERT
- **Embeddings**: 100% match (cosine similarity 1.0)
- **Speed**: 1.55x faster
- **Size**: 569 MB → 143 MB (75% reduction)

See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for detailed results.

## Usage

### Convert a Model

```bash
# Basic conversion
uv run convert --model-id lightonai/Reason-ModernColBERT

# With quantization (recommended)
uv run convert --model-id lightonai/Reason-ModernColBERT --quantize

# Custom output directory
uv run convert \
  --model-id mixedbread-ai/mxbai-edge-colbert-v0-17m \
  --output-dir ./my-models \
  --quantize
```

### Validate a Conversion

```bash
# Basic validation
uv run validate --model-id lightonai/Reason-ModernColBERT

# With custom settings
uv run validate \
  --model-id lightonai/Reason-ModernColBERT \
  --num-runs 50 \
  --output results.json
```

## Tested Models

- ✅ `mixedbread-ai/mxbai-edge-colbert-v0-17m` - Small, fast model (17M params)
- ✅ `lightonai/Reason-ModernColBERT` - Larger, more capable model (150M params)

## Output Structure

```
models/
└── model-id/
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── conversion_metadata.json
    └── onnx/
        ├── model.onnx              # Full precision
        └── model_quantized.onnx    # INT8 quantized
```

## Command Reference

### convert

```bash
uv run convert --model-id <model-id> [options]

Options:
  --model-id          Model ID from HuggingFace (required)
  --output-dir        Output directory (default: ./models/)
  --quantize          Enable INT8 quantization
  --skip-onnxslim     Skip ONNX optimization
  --device            Device: cpu or cuda (default: cpu)
  --opset-version     ONNX opset version (default: 17)
```

### validate

```bash
uv run validate --model-id <model-id> [options]

Options:
  --model-id          Model ID from HuggingFace (required)
  --model-dir         Model directory (default: ./models/<model-id>)
  --num-runs          Number of benchmark runs (default: 10)
  --output            Save results to JSON file
```

## How It Works

1. **Load with PyLate**: Properly loads all ColBERT layers
2. **Detect Architecture**: Automatically identifies transformer + dense layers
3. **Export to ONNX**: Creates ONNX model with dynamic batch/sequence axes
4. **Optimize**: Applies onnxslim for graph optimization
5. **Quantize** (optional): INT8 dynamic quantization for smaller size
6. **Validate**: Compares embeddings, speed, and memory usage

## Requirements

- Python >= 3.12
- PyTorch
- PyLate
- ONNX Runtime
- transformers
- onnxslim
- optimum

See `pyproject.toml` for complete list.

## License

MIT

## Acknowledgments

- [PyLate](https://github.com/lightonai/pylate) - ColBERT model loading
- [transformers.js](https://github.com/xenova/transformers.js) - ONNX inference in JavaScript
- [Optimum](https://github.com/huggingface/optimum) - ONNX utilities
