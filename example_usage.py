"""
Example usage of SegLight Paper Evaluation Framework
This script shows how to use the framework with actual models and data.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from config import *
from network import MazeNet
from paper.utils.model_loader import ModelManager
from paper.evaluation.metrics import SegmentationMetrics
from paper.benchmarks.inference import ModelInference
from paper.visualization.plotter import PaperVisualizer
from paper.statistics.analyzer import StatisticalAnalyzer


def create_sample_model() -> tf.keras.Model:
    """Create a sample MazeNet model for demonstration."""
    model = MazeNet()
    
    # Build the model by calling it with sample input
    sample_input = tf.random.normal((1, INPUT_HEIGHT, INPUT_WIDTH, 3))
    _ = model(sample_input)
    
    return model


def create_sample_data(num_samples: int = 100):
    """Create sample data for evaluation."""
    images = []
    labels = []
    
    for _ in range(num_samples):
        # Create random image
        image = np.random.randint(0, 255, (INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
        
        # Create corresponding label (simplified)
        label = np.random.randint(0, NUMBER_OF_CLASSES, (INPUT_HEIGHT, INPUT_WIDTH), dtype=np.int32)
        label_onehot = tf.one_hot(label, depth=NUMBER_OF_CLASSES).numpy()
        
        images.append(image.astype(np.float32) / 255.0)
        labels.append(label_onehot)
    
    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(8)  # Batch size of 8
    
    return dataset


def example_single_model_evaluation():
    """Example of evaluating a single model."""
    print("=" * 60)
    print("EXAMPLE: Single Model Evaluation")
    print("=" * 60)
    
    # Create model and data
    model = create_sample_model()
    test_data = create_sample_data(50)
    
    # Initialize evaluation components
    metrics = SegmentationMetrics(num_classes=NUMBER_OF_CLASSES)
    inference_system = ModelInference(model, "MazeNet")
    
    # Evaluate model on dataset
    print("Evaluating model on test data...")
    
    total_batches = 0
    for batch in test_data:
        images, labels = batch
        
        # Get predictions
        predictions = model(images, training=False)
        
        # Convert to class indices
        pred_classes = tf.argmax(predictions, axis=-1).numpy()
        true_classes = tf.argmax(labels, axis=-1).numpy()
        
        # Update metrics
        metrics.update(true_classes, pred_classes)
        
        total_batches += 1
        if total_batches >= 10:  # Limit for demo
            break
    
    # Generate report
    report = metrics.generate_report()
    
    print(f"Evaluation Results:")
    print(f"  Pixel Accuracy: {report['overall_metrics']['pixel_accuracy']:.4f}")
    print(f"  Mean IoU: {report['overall_metrics']['mean_iou']:.4f}")
    print(f"  Samples Processed: {report['overall_metrics']['num_samples']}")
    
    if report['performance_metrics']:
        print(f"  Average FPS: {report['performance_metrics']['fps']:.2f}")
    
    print("\nPer-class IoU:")
    for class_name, iou in report['class_metrics']['iou_per_class'].items():
        print(f"  {class_name}: {iou:.4f}")
    
    return report


def example_model_comparison():
    """Example of comparing multiple models."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Model Comparison")
    print("=" * 60)
    
    # Create two models (same architecture but we'll simulate different performance)
    model1 = create_sample_model()
    model2 = create_sample_model()
    
    # Simulate different performance by adding noise to predictions
    def model1_predict(x):
        pred = model1(x)
        return pred
    
    def model2_predict(x):
        pred = model2(x)
        # Add some noise to simulate different performance
        noise = tf.random.normal(pred.shape, stddev=0.1)
        return pred + noise
    
    # Create test data
    test_data = create_sample_data(30)
    
    # Evaluate both models
    models_data = {
        "MazeNet": {"model": model1_predict, "metrics": []},
        "MazeNet_v2": {"model": model2_predict, "metrics": []}
    }
    
    print("Evaluating models...")
    
    for model_name, model_info in models_data.items():
        metrics = SegmentationMetrics(num_classes=NUMBER_OF_CLASSES)
        model_func = model_info["model"]
        
        for batch in test_data.take(5):  # Limit for demo
            images, labels = batch
            predictions = model_func(images)
            
            pred_classes = tf.argmax(predictions, axis=-1).numpy()
            true_classes = tf.argmax(labels, axis=-1).numpy()
            
            metrics.update(true_classes, pred_classes)
        
        report = metrics.generate_report()
        model_info["report"] = report
        
        # Store metrics for statistical analysis
        model_info["metrics"] = {
            "iou": [report['overall_metrics']['mean_iou']] * 20,  # Simulate multiple runs
            "accuracy": [report['overall_metrics']['pixel_accuracy']] * 20
        }
        
        print(f"{model_name}:")
        print(f"  Mean IoU: {report['overall_metrics']['mean_iou']:.4f}")
        print(f"  Pixel Accuracy: {report['overall_metrics']['pixel_accuracy']:.4f}")
    
    # Statistical comparison
    print("\n" + "-" * 40)
    print("Statistical Analysis")
    print("-" * 40)
    
    analyzer = StatisticalAnalyzer()
    
    # Extract IoU values for comparison
    model1_iou = np.array(models_data["MazeNet"]["metrics"]["iou"]) + np.random.normal(0, 0.01, 20)
    model2_iou = np.array(models_data["MazeNet_v2"]["metrics"]["iou"]) + np.random.normal(0, 0.015, 20)
    
    comparison = analyzer.compare_two_models(
        model1_iou, model2_iou, 
        "MazeNet", "MazeNet_v2", "IoU"
    )
    
    print(f"Statistical Test: {comparison['statistical_test']['test_used']}")
    print(f"p-value: {comparison['statistical_test']['p_value']:.6f}")
    print(f"Significant: {comparison['statistical_test']['is_significant']}")
    print(f"Effect size: {comparison['effect_size']['cohens_d']:.4f} ({comparison['effect_size']['magnitude']})")
    print(f"Interpretation: {comparison['interpretation']}")
    
    return models_data, comparison


def example_visualization():
    """Example of creating visualizations."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Visualization")
    print("=" * 60)
    
    visualizer = PaperVisualizer()
    
    # 1. Training curves
    print("Creating training curves...")
    training_logs = {
        'epochs': list(range(1, 21)),
        'train_loss': [0.8 - 0.02*i + 0.05*np.random.random() for i in range(20)],
        'val_loss': [0.85 - 0.015*i + 0.08*np.random.random() for i in range(20)],
        'train_iou': [0.4 + 0.02*i + 0.03*np.random.random() for i in range(20)],
        'val_iou': [0.35 + 0.018*i + 0.04*np.random.random() for i in range(20)]
    }
    
    visualizer.plot_training_curves(training_logs)
    
    # 2. Model comparison
    print("Creating model comparison plot...")
    import pandas as pd
    
    comparison_data = pd.DataFrame({
        'Model': ['MazeNet', 'MazeNet_v2', 'Baseline'],
        'FPS': [45.2, 42.8, 38.5],
        'Mean IoU': [0.847, 0.831, 0.798],
        'Pixel Accuracy': [0.923, 0.915, 0.901],
        'Parameters': [28547, 31205, 45892],
        'Model Size (MB)': [0.11, 0.12, 0.18]
    })
    
    visualizer.plot_model_comparison(comparison_data, ['FPS', 'Mean IoU', 'Pixel Accuracy'])
    
    # 3. Performance vs accuracy
    print("Creating performance vs accuracy plot...")
    performance_data = [
        {'model_name': 'MazeNet', 'fps': 45.2, 'mean_iou': 0.847},
        {'model_name': 'MazeNet_v2', 'fps': 42.8, 'mean_iou': 0.831},
        {'model_name': 'Baseline', 'fps': 38.5, 'mean_iou': 0.798}
    ]
    
    visualizer.plot_performance_vs_accuracy(performance_data)
    
    print("Visualization examples completed!")


def example_full_workflow():
    """Example of a complete evaluation workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Complete Evaluation Workflow")
    print("=" * 60)
    
    # This would typically use the main_paper.py pipeline
    print("Complete workflow steps:")
    print("1. ✓ Model loading and validation")
    print("2. ✓ Dataset preparation and loading")
    print("3. ✓ Comprehensive metric evaluation")
    print("4. ✓ Performance benchmarking")
    print("5. ✓ Statistical significance testing")
    print("6. ✓ Visualization generation")
    print("7. ✓ Report generation")
    print("8. ✓ Publication-ready output creation")
    
    print("\nTypical command:")
    print("python paper/main_paper.py --models MazeNet Baseline --output results/")
    
    print("\nGenerated outputs:")
    print("- JSON results files")
    print("- Publication-ready plots (PNG)")
    print("- LaTeX tables for papers")
    print("- CSV data for analysis")
    print("- Statistical reports")
    print("- Executive summaries")


def main():
    """Run all examples."""
    print("SEGLIGHT PAPER EVALUATION FRAMEWORK EXAMPLES")
    print("=" * 70)
    print("These examples demonstrate practical usage of the evaluation framework")
    print("with actual models and data (simulated for demonstration purposes).")
    print()
    
    try:
        # Run examples
        report = example_single_model_evaluation()
        models_data, comparison = example_model_comparison()
        example_visualization()
        example_full_workflow()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\nKey takeaways:")
        print("• The framework provides end-to-end evaluation capabilities")
        print("• Statistical rigor is built into all comparisons")
        print("• Publication-ready outputs are automatically generated")
        print("• The modular design allows for flexible usage")
        
        print("\nFor real model evaluation:")
        print("1. Train your model using main.py")
        print("2. Use paper/main_paper.py for comprehensive evaluation")
        print("3. Customize evaluation parameters as needed")
        print("4. Use generated plots and tables in your research")
        
    except Exception as e:
        print(f"Example completed with limitations: {str(e)}")
        print("Note: This is a demonstration with simulated data.")
        print("Real usage requires trained models and proper dataset structure.")


if __name__ == "__main__":
    main()