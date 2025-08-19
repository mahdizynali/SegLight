# SegLight Paper Evaluation Framework

A comprehensive evaluation framework for the SegLight semantic segmentation model, designed to support academic publication and research. This framework provides all the tools needed to evaluate, benchmark, and compare semantic segmentation models with statistical rigor and publication-ready outputs.

## Features

### 🔧 **Model Management**
- Support for multiple model formats (TensorFlow, Keras, custom architectures)
- Unified model loading and management interface
- Model metadata and parameter counting
- Easy model comparison setup

### 📊 **Comprehensive Evaluation**
- Semantic segmentation metrics (IoU, mIoU, pixel accuracy)
- Per-class performance analysis
- Confusion matrix generation
- Confidence map evaluation

### ⚡ **Performance Benchmarking**
- Inference time measurement with statistical analysis
- Memory usage profiling
- FPS calculation and stability analysis
- Real-time performance testing
- Cross-platform benchmarking

### 📈 **Statistical Analysis**
- Statistical significance testing (t-tests, Mann-Whitney U, ANOVA, Kruskal-Wallis)
- Effect size calculation (Cohen's d)
- Confidence intervals
- Multiple model comparison with post-hoc analysis
- Performance stability analysis

### 📊 **Publication-Ready Visualizations**
- Training curve plots
- Model comparison charts
- Performance vs accuracy trade-off plots
- Confusion matrices
- Architecture diagrams
- Class-wise performance plots
- Inference time distributions

### 📝 **Automated Report Generation**
- Comprehensive evaluation reports
- LaTeX table generation for papers
- CSV data export for further analysis
- Statistical significance reports
- Executive summaries

## Framework Structure

```
paper/
├── utils/
│   └── model_loader.py          # Model management and loading utilities
├── evaluation/
│   └── metrics.py               # Comprehensive evaluation metrics
├── benchmarks/
│   └── inference.py             # Performance benchmarking and inference
├── visualization/
│   └── plotter.py               # Publication-ready plots and figures
├── statistics/
│   └── analyzer.py              # Statistical analysis tools
├── reports/                     # Generated evaluation reports
├── models/                      # Model storage directory
└── main_paper.py               # Main evaluation pipeline
```

## Quick Start

### 1. Basic Demo
```bash
# Run the demonstration to see framework capabilities
python simple_demo.py
```

### 2. Single Model Evaluation
```bash
# Evaluate a single model
python paper/main_paper.py --single MazeNet --output results/
```

### 3. Multiple Model Comparison
```bash
# Compare multiple models
python paper/main_paper.py --models MazeNet Baseline --output results/
```

### 4. Custom Evaluation
```python
from paper import PaperEvaluationPipeline

# Create evaluation pipeline
pipeline = PaperEvaluationPipeline(output_dir="my_results")

# Run full evaluation
results_dir = pipeline.run_full_evaluation(["MazeNet", "Baseline"])
```

## Detailed Usage

### Model Management

```python
from paper.utils.model_loader import ModelManager, create_default_models

# Create model manager
manager = create_default_models()

# Register a new model
manager.register_model(
    name="MyModel",
    model_path="path/to/model",
    model_type="tensorflow",
    description="Custom semantic segmentation model"
)

# Load and get model summary
model = manager.load_model("MyModel")
summary = manager.get_model_summary("MyModel")
```

### Evaluation Metrics

```python
from paper.evaluation.metrics import SegmentationMetrics, evaluate_model_on_dataset

# Initialize metrics
metrics = SegmentationMetrics(num_classes=3)

# Update with predictions
metrics.update(ground_truth, predictions, inference_time)

# Generate comprehensive report
report = metrics.generate_report()
```

### Performance Benchmarking

```python
from paper.benchmarks.inference import ModelInference, benchmark_multiple_models

# Create inference system
inference_system = ModelInference(model, "MyModel")

# Single image inference with timing
result = inference_system.single_image_inference(image)

# Benchmark inference speed
benchmark_results = inference_system.benchmark_inference_speed(
    image, num_iterations=100
)
```

### Statistical Analysis

```python
from paper.statistics.analyzer import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Compare two models
comparison = analyzer.compare_two_models(
    model1_metrics, model2_metrics, 
    "Model1", "Model2", "IoU"
)

# Multiple model comparison
multi_comparison = analyzer.multiple_model_comparison(
    {"Model1": metrics1, "Model2": metrics2, "Model3": metrics3},
    "Accuracy"
)
```

### Visualization

```python
from paper.visualization.plotter import PaperVisualizer

visualizer = PaperVisualizer()

# Plot training curves
visualizer.plot_training_curves(training_logs)

# Model comparison
visualizer.plot_model_comparison(comparison_data)

# Architecture diagram
visualizer.create_architecture_diagram()
```

## Output Files

The evaluation pipeline generates several types of outputs:

### Reports and Data
- `model_comparison_results.json` - Complete evaluation results
- `evaluation_summary.txt` - Human-readable summary
- `results_data.csv` - Raw data for analysis
- `model_comparison_table.tex` - LaTeX table for papers

### Visualizations
- `training_curves.png` - Training progress plots
- `model_comparison.png` - Model performance comparison
- `performance_vs_accuracy.png` - Trade-off analysis
- `confusion_matrix.png` - Classification confusion matrix
- `mazenet_architecture.png` - Architecture diagram

### Individual Model Results
- `{ModelName}_results.json` - Detailed model evaluation

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Pandas
- SciPy
- Scikit-learn
- psutil
- tqdm

Install requirements:
```bash
pip install tensorflow opencv-python numpy matplotlib seaborn pandas scipy scikit-learn psutil tqdm
```

## Research Applications

This framework is designed to support academic research in semantic segmentation:

### Academic Papers
- Generate publication-ready figures and tables
- Provide statistical significance testing
- Create comprehensive evaluation reports
- Support reproducible research

### Model Development
- Compare different architectures
- Analyze performance trade-offs
- Identify optimization opportunities
- Validate improvements

### Benchmarking Studies
- Standardized evaluation protocols
- Statistical rigor in comparisons
- Cross-platform performance analysis
- Reproducible benchmark results

## Key Metrics Supported

### Segmentation Metrics
- **Mean Intersection over Union (mIoU)**
- **Pixel Accuracy**
- **Per-class IoU**
- **Per-class Accuracy**
- **Confusion Matrix**

### Performance Metrics
- **Inference Time** (mean, std, percentiles)
- **Frames Per Second (FPS)**
- **Memory Usage** (RSS, VMS, delta)
- **Model Size** (parameters, MB)
- **Throughput** (images/second)

### Statistical Tests
- **T-tests** for normal distributions
- **Mann-Whitney U** for non-normal distributions
- **ANOVA** for multiple group comparisons
- **Kruskal-Wallis** for non-parametric multiple comparisons
- **Effect Size** calculation (Cohen's d)

## Best Practices

### Model Evaluation
1. **Use sufficient test data** - Ensure statistical power
2. **Multiple runs** - Account for variance in performance
3. **Controlled conditions** - Consistent evaluation environment
4. **Cross-validation** - When applicable for robust estimates

### Statistical Analysis
1. **Check normality** - Use appropriate tests
2. **Multiple comparisons** - Apply corrections when needed
3. **Effect sizes** - Report practical significance
4. **Confidence intervals** - Provide uncertainty estimates

### Publication
1. **Reproducibility** - Save all parameters and settings
2. **Transparency** - Report all evaluation details
3. **Statistical rigor** - Use proper statistical methods
4. **Clear presentation** - Use generated plots and tables

## Examples

See the `simple_demo.py` file for comprehensive examples of all framework features.

## Citation

If you use this evaluation framework in your research, please cite:

```bibtex
@software{SegLight_Evaluation_Framework,
  title={SegLight Paper Evaluation Framework},
  author={SegLight Research Team},
  year={2024},
  url={https://github.com/mahdizynali/SegLight}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## License

This project is licensed under the same license as the main SegLight project.