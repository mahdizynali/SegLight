"""
SegLight Paper Evaluation Framework

This package provides comprehensive evaluation tools for the SegLight semantic segmentation model,
designed to support academic publication and research.

Modules:
- utils: Model loading and management utilities
- evaluation: Comprehensive evaluation metrics for semantic segmentation
- benchmarks: Performance benchmarking and inference testing
- visualization: Publication-ready plotting and visualization
- statistics: Statistical analysis and significance testing
"""

__version__ = "1.0.0"
__author__ = "SegLight Research Team"

from .utils.model_loader import ModelManager, create_default_models
from .evaluation.metrics import SegmentationMetrics, evaluate_model_on_dataset
from .benchmarks.inference import ModelInference, benchmark_multiple_models
from .visualization.plotter import PaperVisualizer, create_publication_plots
from .statistics.analyzer import StatisticalAnalyzer, BenchmarkStatistics
from .main_paper import PaperEvaluationPipeline

__all__ = [
    'ModelManager',
    'create_default_models',
    'SegmentationMetrics',
    'evaluate_model_on_dataset',
    'ModelInference',
    'benchmark_multiple_models',
    'PaperVisualizer',
    'create_publication_plots',
    'StatisticalAnalyzer',
    'BenchmarkStatistics',
    'PaperEvaluationPipeline'
]