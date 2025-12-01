#!/usr/bin/env python3
"""
Validation script for ColBERT ONNX models.
Compares embeddings, speed, and memory usage between PyTorch and ONNX.
"""

import argparse
import time
import os
from typing import List, Tuple, Dict
import json
import tempfile

import torch
import numpy as np
import onnxruntime as ort
from pylate import models
from transformers import AutoTokenizer
from scipy.spatial.distance import cosine
import psutil


class ValidationResults:
    """Store and display validation results."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.embedding_similarity = {}
        self.speed_comparison = {}
        self.memory_comparison = {}
        self.errors = []

    def add_embedding_similarity(self, model_type: str, metrics: Dict):
        self.embedding_similarity[model_type] = metrics

    def add_speed_comparison(self, model_type: str, metrics: Dict):
        self.speed_comparison[model_type] = metrics

    def add_memory_comparison(self, model_type: str, metrics: Dict):
        self.memory_comparison[model_type] = metrics

    def add_error(self, error: str):
        self.errors.append(error)

    def print_report(self):
        """Print a comprehensive validation report."""
        print("\n" + "=" * 80)
        print(f"VALIDATION REPORT: {self.model_id}")
        print("=" * 80)

        # Embedding Similarity
        print("\n📊 EMBEDDING SIMILARITY")
        print("-" * 80)
        for model_type, metrics in self.embedding_similarity.items():
            print(f"\n{model_type.upper()}:")
            print(f"  Cosine Similarity:  {metrics['cosine_similarity']:.6f}")
            print(f"  MSE:                {metrics['mse']:.8f}")
            print(f"  Max Absolute Diff:  {metrics['max_abs_diff']:.8f}")
            print(f"  Mean Absolute Diff: {metrics['mean_abs_diff']:.8f}")

            # Status
            if metrics['cosine_similarity'] >= 0.99:
                print(f"  ✅ PASS - Embeddings are highly similar!")
            elif metrics['cosine_similarity'] >= 0.95:
                print(f"  ⚠️  WARN - Embeddings are similar but not identical")
            else:
                print(f"  ❌ FAIL - Embeddings differ significantly")

        # Speed Comparison
        print("\n⚡ SPEED COMPARISON")
        print("-" * 80)
        if self.speed_comparison:
            pytorch_time = self.speed_comparison.get('pytorch', {}).get('avg_time', 0)

            for model_type, metrics in self.speed_comparison.items():
                if model_type == 'pytorch':
                    print(f"\n{model_type.upper()} (baseline):")
                    print(f"  Average Time: {metrics['avg_time']*1000:.2f} ms")
                    print(f"  Total Time:   {metrics['total_time']:.2f} s")
                else:
                    speedup = pytorch_time / metrics['avg_time'] if metrics['avg_time'] > 0 else 0
                    print(f"\n{model_type.upper()}:")
                    print(f"  Average Time: {metrics['avg_time']*1000:.2f} ms")
                    print(f"  Total Time:   {metrics['total_time']:.2f} s")
                    print(f"  Speedup:      {speedup:.2f}x faster than PyTorch")

        # Memory Comparison
        print("\n💾 MEMORY COMPARISON")
        print("-" * 80)
        for model_type, metrics in self.memory_comparison.items():
            print(f"\n{model_type.upper()}:")
            print(f"  Model Size:      {metrics['model_size_mb']:.2f} MB")
            print(f"  Peak Memory:     {metrics['peak_memory_mb']:.2f} MB")

            if model_type != 'pytorch':
                pytorch_size = self.memory_comparison.get('pytorch', {}).get('model_size_mb', 0)
                if pytorch_size > 0:
                    reduction = (1 - metrics['model_size_mb'] / pytorch_size) * 100
                    print(f"  Size Reduction:  {reduction:.1f}%")

        # Errors
        if self.errors:
            print("\n❌ ERRORS")
            print("-" * 80)
            for error in self.errors:
                print(f"  • {error}")

        print("\n" + "=" * 80)

    def to_dict(self) -> Dict:
        """Export results as dictionary."""
        return {
            "model_id": self.model_id,
            "embedding_similarity": self.embedding_similarity,
            "speed_comparison": self.speed_comparison,
            "memory_comparison": self.memory_comparison,
            "errors": self.errors,
        }


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_model_disk_size_mb(model: torch.nn.Module) -> float:
    """Persist state_dict to temp file to measure on-disk model size."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        torch.save(model.state_dict(), tmp_path)
        return os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def compute_embedding_similarity(
    pytorch_embeddings: np.ndarray,
    onnx_embeddings: np.ndarray
) -> Dict:
    """
    Compute similarity metrics between PyTorch and ONNX embeddings.

    Args:
        pytorch_embeddings: PyTorch model embeddings
        onnx_embeddings: ONNX model embeddings

    Returns:
        Dictionary of similarity metrics
    """
    # Flatten for cosine similarity
    pytorch_flat = pytorch_embeddings.flatten()
    onnx_flat = onnx_embeddings.flatten()

    # Compute metrics
    cosine_sim = 1 - cosine(pytorch_flat, onnx_flat)
    mse = np.mean((pytorch_embeddings - onnx_embeddings) ** 2)
    max_abs_diff = np.max(np.abs(pytorch_embeddings - onnx_embeddings))
    mean_abs_diff = np.mean(np.abs(pytorch_embeddings - onnx_embeddings))

    return {
        "cosine_similarity": float(cosine_sim),
        "mse": float(mse),
        "max_abs_diff": float(max_abs_diff),
        "mean_abs_diff": float(mean_abs_diff),
    }


def benchmark_pytorch_model(
    model,
    tokenizer,
    test_texts: List[str],
    num_runs: int = 10
) -> Tuple[np.ndarray, Dict]:
    """
    Benchmark PyTorch model.

    Args:
        model: PyLate ColBERT model
        tokenizer: Tokenizer
        test_texts: List of test texts
        num_runs: Number of benchmark runs

    Returns:
        Tuple of (embeddings, metrics)
    """
    print(f"\nBenchmarking PyTorch model ({num_runs} runs)...")

    # Get model size (on-disk)
    model_size_mb = get_model_disk_size_mb(model)

    # Tokenize with fixed length for fair comparison
    inputs = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # Extract the modules from PyLate model
    modules = list(model.named_children())
    module_list = [module for name, module in modules]

    # Create a forward function that mimics the ONNX wrapper
    def forward_pytorch(input_ids, attention_mask):
        features = module_list[0]({"input_ids": input_ids, "attention_mask": attention_mask})
        embeddings = features["token_embeddings"]

        for module in module_list[1:]:
            features = module({"token_embeddings": embeddings})
            embeddings = features["token_embeddings"]

        return embeddings

    # Warmup
    with torch.no_grad():
        _ = forward_pytorch(inputs["input_ids"][:1], inputs["attention_mask"][:1])

    # Benchmark
    times = []
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    peak_rss = mem_before

    for i in range(num_runs):
        start = time.time()
        with torch.no_grad():
            embeddings = forward_pytorch(inputs["input_ids"], inputs["attention_mask"])
        times.append(time.time() - start)
        peak_rss = max(peak_rss, process.memory_info().rss)

    mem_peak_mb = (peak_rss - mem_before) / (1024 * 1024)

    # Get final embeddings for comparison
    with torch.no_grad():
        final_embeddings = forward_pytorch(inputs["input_ids"], inputs["attention_mask"])

    # Convert to numpy array
    if isinstance(final_embeddings, torch.Tensor):
        final_embeddings = final_embeddings.cpu().numpy()

    metrics = {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "total_time": sum(times),
        "model_size_mb": model_size_mb,
        "peak_memory_mb": mem_peak_mb,
    }

    print(f"  ✓ Average time: {metrics['avg_time']*1000:.2f} ms")

    return final_embeddings, metrics


def benchmark_onnx_model(
    onnx_path: str,
    tokenizer,
    test_texts: List[str],
    num_runs: int = 10
) -> Tuple[np.ndarray, Dict]:
    """
    Benchmark ONNX model.

    Args:
        onnx_path: Path to ONNX model
        tokenizer: Tokenizer
        test_texts: List of test texts
        num_runs: Number of benchmark runs

    Returns:
        Tuple of (embeddings, metrics)
    """
    model_name = os.path.basename(onnx_path)
    print(f"\nBenchmarking ONNX model: {model_name} ({num_runs} runs)...")

    # Get model size
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    # Create session
    session = ort.InferenceSession(onnx_path)

    # Tokenize with same parameters as PyTorch
    inputs = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="np"
    )

    # Warmup
    _ = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )

    # Benchmark
    times = []
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    peak_rss = mem_before

    for i in range(num_runs):
        start = time.time()
        outputs = session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        )
        times.append(time.time() - start)
        peak_rss = max(peak_rss, process.memory_info().rss)

    mem_peak_mb = (peak_rss - mem_before) / (1024 * 1024)

    # Get final embeddings
    final_outputs = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )

    embeddings = final_outputs[0]

    metrics = {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "total_time": sum(times),
        "model_size_mb": model_size_mb,
        "peak_memory_mb": mem_peak_mb,
    }

    print(f"  ✓ Average time: {metrics['avg_time']*1000:.2f} ms")

    return embeddings, metrics


def validate_model(
    model_id: str,
    model_dir: str,
    test_texts: List[str],
    num_runs: int = 10
) -> ValidationResults:
    """
    Validate a converted ColBERT model.

    Args:
        model_id: HuggingFace model ID
        model_dir: Directory containing converted model
        test_texts: Test texts for validation
        num_runs: Number of benchmark runs

    Returns:
        ValidationResults object
    """
    results = ValidationResults(model_id)

    print("\n" + "=" * 80)
    print(f"VALIDATING: {model_id}")
    print("=" * 80)

    try:
        # Load PyTorch model
        print("\nLoading PyTorch model...")
        pytorch_model = models.ColBERT(model_name_or_path=model_id, device='cpu')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("  ✓ Loaded")

        # Benchmark PyTorch
        pytorch_embeddings, pytorch_metrics = benchmark_pytorch_model(
            pytorch_model, tokenizer, test_texts, num_runs
        )
        results.add_speed_comparison('pytorch', pytorch_metrics)
        results.add_memory_comparison('pytorch', {
            'model_size_mb': pytorch_metrics['model_size_mb'],
            'peak_memory_mb': pytorch_metrics['peak_memory_mb']
        })

        # Validate ONNX models
        onnx_dir = os.path.join(model_dir, 'onnx')

        # Regular ONNX model
        regular_onnx_path = os.path.join(onnx_dir, 'model.onnx')
        if os.path.exists(regular_onnx_path):
            onnx_embeddings, onnx_metrics = benchmark_onnx_model(
                regular_onnx_path, tokenizer, test_texts, num_runs
            )

            # Compare embeddings
            similarity_metrics = compute_embedding_similarity(
                pytorch_embeddings, onnx_embeddings
            )
            results.add_embedding_similarity('onnx', similarity_metrics)
            results.add_speed_comparison('onnx', onnx_metrics)
            results.add_memory_comparison('onnx', {
                'model_size_mb': onnx_metrics['model_size_mb'],
                'peak_memory_mb': onnx_metrics['peak_memory_mb']
            })
        else:
            results.add_error(f"ONNX model not found: {regular_onnx_path}")

        # Quantized ONNX model
        quantized_onnx_path = os.path.join(onnx_dir, 'model_quantized.onnx')
        if os.path.exists(quantized_onnx_path):
            quant_embeddings, quant_metrics = benchmark_onnx_model(
                quantized_onnx_path, tokenizer, test_texts, num_runs
            )

            # Compare embeddings
            similarity_metrics = compute_embedding_similarity(
                pytorch_embeddings, quant_embeddings
            )
            results.add_embedding_similarity('quantized', similarity_metrics)
            results.add_speed_comparison('quantized', quant_metrics)
            results.add_memory_comparison('quantized', {
                'model_size_mb': quant_metrics['model_size_mb'],
                'peak_memory_mb': quant_metrics['peak_memory_mb']
            })
        else:
            print(f"\n  ⚠️  Quantized model not found: {quantized_onnx_path}")

    except Exception as e:
        results.add_error(f"Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate ColBERT ONNX models"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model identifier"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Directory containing converted model (default: ./models/<model-id>)"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of benchmark runs (default: 10)"
    )
    parser.add_argument(
        "--test-texts",
        type=str,
        nargs="+",
        default=[
            "What is the capital of France?",
            "How does machine learning work?",
            "Explain quantum computing in simple terms.",
            "What are the benefits of exercise?",
            "How to make a chocolate cake?",
        ],
        help="Test texts for validation"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Determine model directory
    if args.model_dir is None:
        args.model_dir = os.path.join("./models", args.model_id)

    if not os.path.exists(args.model_dir):
        print(f"❌ Error: Model directory not found: {args.model_dir}")
        print(f"   Run conversion first: uv run convert --model-id {args.model_id}")
        return 1

    # Run validation
    results = validate_model(
        args.model_id,
        args.model_dir,
        args.test_texts,
        args.num_runs
    )

    # Print report
    results.print_report()

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
