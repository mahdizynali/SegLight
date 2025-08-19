"""
Comprehensive evaluation metrics for semantic segmentation models.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import COLOR_MAP, NUMBER_OF_CLASSES


class SegmentationMetrics:
    """
    Comprehensive metrics calculation for semantic segmentation.
    """
    
    def __init__(self, num_classes: int = NUMBER_OF_CLASSES, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or list(COLOR_MAP.keys())
        self.reset()
        
    def reset(self):
        """Reset all accumulated metrics."""
        self.total_cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.pixel_accuracies = []
        self.mean_ious = []
        self.inference_times = []
        self.predictions_list = []
        self.labels_list = []
        
    def update(self, y_true: np.ndarray, y_pred: np.ndarray, inference_time: float = None):
        """
        Update metrics with new predictions.
        
        Args:
            y_true: Ground truth labels (H, W) or (B, H, W)
            y_pred: Predicted labels (H, W) or (B, H, W)
            inference_time: Time taken for inference in seconds
        """
        # Ensure we have the right shape
        if len(y_true.shape) == 3:  # Batch dimension
            for i in range(y_true.shape[0]):
                self._update_single(y_true[i], y_pred[i])
        else:
            self._update_single(y_true, y_pred)
            
        if inference_time is not None:
            self.inference_times.append(inference_time)
            
    def _update_single(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Update metrics for a single image."""
        # Flatten arrays
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Update confusion matrix
        cm = confusion_matrix(y_true_flat, y_pred_flat, labels=range(self.num_classes))
        self.total_cm += cm
        
        # Store for later analysis
        self.predictions_list.append(y_pred_flat)
        self.labels_list.append(y_true_flat)
        
        # Calculate pixel accuracy for this image
        pixel_acc = np.sum(y_true_flat == y_pred_flat) / len(y_true_flat)
        self.pixel_accuracies.append(pixel_acc)
        
        # Calculate IoU for this image
        iou_per_class = []
        for class_id in range(self.num_classes):
            intersection = np.sum((y_true_flat == class_id) & (y_pred_flat == class_id))
            union = np.sum((y_true_flat == class_id) | (y_pred_flat == class_id))
            if union > 0:
                iou_per_class.append(intersection / union)
            else:
                iou_per_class.append(1.0)  # Perfect score if class not present
                
        self.mean_ious.append(np.mean(iou_per_class))
        
    def get_pixel_accuracy(self) -> float:
        """Calculate overall pixel accuracy."""
        if len(self.pixel_accuracies) == 0:
            return 0.0
        return np.mean(self.pixel_accuracies)
        
    def get_mean_iou(self) -> float:
        """Calculate mean IoU across all images."""
        if len(self.mean_ious) == 0:
            return 0.0
        return np.mean(self.mean_ious)
        
    def get_class_iou(self) -> Dict[str, float]:
        """Calculate IoU for each class."""
        iou_per_class = {}
        
        for class_id in range(self.num_classes):
            intersection = self.total_cm[class_id, class_id]
            union = (self.total_cm[class_id, :].sum() + 
                    self.total_cm[:, class_id].sum() - 
                    self.total_cm[class_id, class_id])
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0
                
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            iou_per_class[class_name] = iou
            
        return iou_per_class
        
    def get_class_accuracy(self) -> Dict[str, float]:
        """Calculate accuracy for each class."""
        class_acc = {}
        
        for class_id in range(self.num_classes):
            true_positive = self.total_cm[class_id, class_id]
            total_true = self.total_cm[class_id, :].sum()
            
            if total_true > 0:
                acc = true_positive / total_true
            else:
                acc = 0.0
                
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class_{class_id}"
            class_acc[class_name] = acc
            
        return class_acc
        
    def get_inference_stats(self) -> Dict[str, float]:
        """Get inference time statistics."""
        if not self.inference_times:
            return {}
            
        times = np.array(self.inference_times)
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'fps': 1.0 / np.mean(times) if np.mean(times) > 0 else 0.0
        }
        
    def get_confusion_matrix(self) -> np.ndarray:
        """Get normalized confusion matrix."""
        return self.total_cm.astype('float') / self.total_cm.sum(axis=1)[:, np.newaxis]
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        report = {
            'overall_metrics': {
                'pixel_accuracy': self.get_pixel_accuracy(),
                'mean_iou': self.get_mean_iou(),
                'num_samples': len(self.pixel_accuracies)
            },
            'class_metrics': {
                'iou_per_class': self.get_class_iou(),
                'accuracy_per_class': self.get_class_accuracy()
            },
            'performance_metrics': self.get_inference_stats(),
            'confusion_matrix': self.total_cm.tolist()
        }
        
        return report
        
    def save_report(self, filepath: str):
        """Save evaluation report to JSON file."""
        import json
        report = self.generate_report()
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
            
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=convert_numpy)


class PerformanceBenchmark:
    """
    Benchmarking utilities for model performance evaluation.
    """
    
    def __init__(self):
        self.results = {}
        
    def benchmark_model(self, model, test_data, model_name: str, 
                       num_warmup: int = 5, num_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark model performance on test data.
        
        Args:
            model: TensorFlow model
            test_data: Test dataset
            model_name: Name identifier for the model
            num_warmup: Number of warmup iterations
            num_iterations: Number of benchmark iterations
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"Benchmarking {model_name}...")
        
        # Get a sample batch for benchmarking
        sample_batch = None
        for batch in test_data.take(1):
            sample_batch = batch
            break
            
        if sample_batch is None:
            raise ValueError("No data available for benchmarking")
            
        images, labels = sample_batch
        
        # Warmup runs
        print(f"Performing {num_warmup} warmup iterations...")
        for _ in range(num_warmup):
            _ = model(images, training=False)
            
        # Benchmark runs
        print(f"Running {num_iterations} benchmark iterations...")
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            predictions = model(images, training=False)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_iterations} iterations")
        
        # Calculate statistics
        times = np.array(times)
        batch_size = images.shape[0] if len(images.shape) > 0 else 1
        
        results = {
            'model_name': model_name,
            'batch_size': batch_size,
            'total_iterations': num_iterations,
            'mean_batch_time': np.mean(times),
            'std_batch_time': np.std(times),
            'min_batch_time': np.min(times),
            'max_batch_time': np.max(times),
            'median_batch_time': np.median(times),
            'mean_image_time': np.mean(times) / batch_size,
            'fps': batch_size / np.mean(times),
            'images_per_second': batch_size / np.mean(times)
        }
        
        self.results[model_name] = results
        return results
        
    def compare_models(self, models_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare benchmark results across multiple models.
        
        Args:
            models_results: List of benchmark results
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for result in models_results:
            comparison_data.append({
                'Model': result['model_name'],
                'FPS': result['fps'],
                'Mean Time (ms)': result['mean_batch_time'] * 1000,
                'Std Time (ms)': result['std_batch_time'] * 1000,
                'Min Time (ms)': result['min_batch_time'] * 1000,
                'Max Time (ms)': result['max_batch_time'] * 1000,
                'Images/sec': result['images_per_second']
            })
            
        return pd.DataFrame(comparison_data)
        
    def save_benchmark_results(self, filepath: str):
        """Save benchmark results to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)


def evaluate_model_on_dataset(model, dataset, model_name: str = "Model") -> SegmentationMetrics:
    """
    Evaluate a model on a complete dataset.
    
    Args:
        model: TensorFlow model
        dataset: Test dataset
        model_name: Name of the model
        
    Returns:
        SegmentationMetrics object with results
    """
    metrics = SegmentationMetrics()
    
    print(f"Evaluating {model_name} on dataset...")
    
    total_batches = 0
    processed_images = 0
    
    for batch in dataset:
        images, labels = batch
        
        # Inference with timing
        start_time = time.time()
        predictions = model(images, training=False)
        inference_time = time.time() - start_time
        
        # Convert predictions and labels to class indices
        pred_classes = tf.argmax(predictions, axis=-1).numpy()
        true_classes = tf.argmax(labels, axis=-1).numpy()
        
        # Update metrics
        metrics.update(true_classes, pred_classes, inference_time)
        
        total_batches += 1
        processed_images += images.shape[0]
        
        if total_batches % 10 == 0:
            print(f"Processed {total_batches} batches, {processed_images} images")
    
    print(f"Evaluation complete. Processed {processed_images} images in {total_batches} batches")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("Segmentation Metrics Example")
    
    # Create dummy data for testing
    num_samples = 100
    height, width = 240, 320
    num_classes = 3
    
    # Simulate some predictions and ground truth
    y_true = np.random.randint(0, num_classes, (num_samples, height, width))
    y_pred = np.random.randint(0, num_classes, (num_samples, height, width))
    
    # Initialize metrics
    metrics = SegmentationMetrics(num_classes=num_classes)
    
    # Update with dummy data
    for i in range(num_samples):
        metrics.update(y_true[i], y_pred[i], inference_time=0.01 + np.random.normal(0, 0.001))
    
    # Generate report
    report = metrics.generate_report()
    print(f"Pixel Accuracy: {report['overall_metrics']['pixel_accuracy']:.4f}")
    print(f"Mean IoU: {report['overall_metrics']['mean_iou']:.4f}")
    print(f"Mean FPS: {report['performance_metrics']['fps']:.2f}")