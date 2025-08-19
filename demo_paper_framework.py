"""
Demo script for SegLight Paper Evaluation Framework
This script demonstrates the key features of the evaluation framework.
"""

import os
import sys
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from paper.utils.model_loader import create_default_models
from paper.evaluation.metrics import SegmentationMetrics
from paper.visualization.plotter import PaperVisualizer
from paper.statistics.analyzer import StatisticalAnalyzer


def demo_model_manager():
    """Demonstrate model management capabilities."""
    print("="*50)
    print("DEMO: Model Manager")
    print("="*50)
    
    # Create model manager
    manager = create_default_models()
    
    print("Available models:")
    for model_name in manager.list_models():
        print(f"- {model_name}")
        info = manager.get_model_info(model_name)
        print(f"  Description: {info['description']}")
        print(f"  Type: {info['type']}")
    
    print()


def demo_metrics():
    """Demonstrate metrics calculation."""
    print("="*50)
    print("DEMO: Segmentation Metrics")
    print("="*50)
    
    # Create sample data
    num_samples = 50
    height, width = 240, 320
    num_classes = 3
    
    # Simulate ground truth and predictions
    y_true = np.random.randint(0, num_classes, (num_samples, height, width))
    # Make predictions somewhat correlated with ground truth
    y_pred = np.copy(y_true)
    # Add some noise
    noise_mask = np.random.random((num_samples, height, width)) < 0.1
    y_pred[noise_mask] = np.random.randint(0, num_classes, np.sum(noise_mask))
    
    # Initialize metrics
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    # Update with sample data
    for i in range(num_samples):
        inference_time = 0.015 + np.random.normal(0, 0.002)  # ~15ms ± 2ms
        metrics.update(y_true[i], y_pred[i], inference_time)
    
    # Generate report
    report = metrics.generate_report()
    
    print(f"Samples processed: {report['overall_metrics']['num_samples']}")
    print(f"Pixel Accuracy: {report['overall_metrics']['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {report['overall_metrics']['mean_iou']:.4f}")
    print(f"Average FPS: {report['performance_metrics']['fps']:.2f}")
    
    print("\nPer-class IoU:")
    for class_name, iou in report['class_metrics']['iou_per_class'].items():
        print(f"  {class_name}: {iou:.4f}")
    
    print()


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("="*50)
    print("DEMO: Visualization")
    print("="*50)
    
    visualizer = PaperVisualizer()
    
    # Create sample training data
    epochs = list(range(1, 51))
    train_loss = [0.8 - 0.01*i + 0.05*np.random.random() for i in range(50)]
    val_loss = [0.85 - 0.008*i + 0.08*np.random.random() for i in range(50)]
    train_iou = [0.3 + 0.01*i + 0.02*np.random.random() for i in range(50)]
    val_iou = [0.25 + 0.008*i + 0.03*np.random.random() for i in range(50)]
    
    training_logs = {
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_iou': train_iou,
        'val_iou': val_iou
    }
    
    print("Creating training curves plot...")
    visualizer.plot_training_curves(training_logs)
    
    print("Creating architecture diagram...")
    visualizer.create_architecture_diagram()
    
    print("Visualization demo completed!")
    print()


def demo_statistics():
    """Demonstrate statistical analysis."""
    print("="*50)
    print("DEMO: Statistical Analysis")
    print("="*50)
    
    analyzer = StatisticalAnalyzer()
    
    # Create sample data for two models
    np.random.seed(42)
    mazenet_iou = np.random.normal(0.85, 0.03, 100)  # MazeNet IoU scores
    baseline_iou = np.random.normal(0.82, 0.04, 100)  # Baseline IoU scores
    
    # Compare models
    comparison = analyzer.compare_two_models(
        mazenet_iou, baseline_iou, 
        "MazeNet", "Baseline", "IoU"
    )
    
    print("Statistical Comparison Results:")
    print(f"MazeNet mean IoU: {comparison['model1_stats']['mean']:.4f} ± {comparison['model1_stats']['std']:.4f}")
    print(f"Baseline mean IoU: {comparison['model2_stats']['mean']:.4f} ± {comparison['model2_stats']['std']:.4f}")
    print(f"Test used: {comparison['statistical_test']['test_used']}")
    print(f"p-value: {comparison['statistical_test']['p_value']:.6f}")
    print(f"Statistically significant: {comparison['statistical_test']['is_significant']}")
    print(f"Effect size (Cohen's d): {comparison['effect_size']['cohens_d']:.4f} ({comparison['effect_size']['magnitude']})")
    print(f"Interpretation: {comparison['interpretation']}")
    
    print()


def demo_full_pipeline():
    """Demonstrate the complete evaluation pipeline concept."""
    print("="*50)
    print("DEMO: Full Pipeline Concept")
    print("="*50)
    
    print("The complete pipeline would include:")
    print("1. ✓ Model loading and management")
    print("2. ✓ Comprehensive metrics evaluation")
    print("3. ✓ Performance benchmarking")
    print("4. ✓ Statistical significance testing")
    print("5. ✓ Publication-ready visualizations")
    print("6. ✓ Automated report generation")
    print("7. ✓ LaTeX table generation")
    print("8. ✓ CSV data export")
    
    print("\nTo run the full pipeline on actual models:")
    print("python paper/main_paper.py --models MazeNet")
    print("python paper/main_paper.py --models MazeNet Baseline --output results/")
    
    print()


def main():
    """Run all demos."""
    print("SEGLIGHT PAPER EVALUATION FRAMEWORK DEMO")
    print("="*60)
    print("This demo showcases the capabilities of the evaluation framework")
    print("developed for academic publication and model comparison.")
    print()
    
    try:
        demo_model_manager()
        demo_metrics()
        demo_statistics()
        demo_visualization()
        demo_full_pipeline()
        
        print("="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("The framework is ready for:")
        print("• Comprehensive model evaluation")
        print("• Statistical analysis and significance testing")
        print("• Publication-ready plots and tables")
        print("• Academic paper support")
        print()
        print("Next steps:")
        print("1. Train your MazeNet model using main.py")
        print("2. Run the evaluation pipeline: python paper/main_paper.py")
        print("3. Use generated plots and tables in your paper")
        
    except Exception as e:
        print(f"Demo error: {str(e)}")
        print("Note: Some features require actual trained models to be fully functional.")


if __name__ == "__main__":
    main()