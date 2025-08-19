"""
Comprehensive inference and benchmarking system for SegLight models.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import psutil
import threading
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import *
from paper.utils.model_loader import ModelManager
from paper.evaluation.metrics import SegmentationMetrics, PerformanceBenchmark


@dataclass
class InferenceResult:
    """Container for inference results."""
    prediction: np.ndarray
    inference_time: float
    memory_usage: Dict[str, float]
    confidence_map: Optional[np.ndarray] = None


class ModelInference:
    """
    Advanced inference system with performance monitoring.
    """
    
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.color_map = COLOR_MAP
        self.class_names = list(COLOR_MAP.keys())
        
        # Create color lookup for visualization
        self.color_lookup_bgr = np.zeros((len(COLOR_MAP), 3), dtype=np.uint8)
        for idx, (class_name, color) in enumerate(COLOR_MAP.items()):
            color_bgr = [color[2], color[1], color[0]]  # RGB to BGR
            self.color_lookup_bgr[idx] = np.array(color_bgr, dtype=np.uint8)
    
    def preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        image_resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
        
        # Convert to float and normalize
        image_tensor = tf.convert_to_tensor(image_resized, dtype=tf.float32)
        image_tensor = image_tensor / 255.0
        
        # Add batch dimension
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        
        return image_tensor
    
    def postprocess_prediction(self, prediction: tf.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess model prediction.
        
        Args:
            prediction: Model output tensor
            
        Returns:
            Tuple of (class_prediction, confidence_map)
        """
        # Get class predictions
        class_pred = tf.argmax(prediction, axis=-1)
        class_pred = class_pred[0].numpy()  # Remove batch dimension
        
        # Get confidence map (max probability)
        confidence = tf.reduce_max(prediction, axis=-1)
        confidence = confidence[0].numpy()  # Remove batch dimension
        
        return class_pred, confidence
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            'percent': process.memory_percent()
        }
    
    def single_image_inference(self, image: np.ndarray, 
                             return_confidence: bool = True) -> InferenceResult:
        """
        Perform inference on a single image.
        
        Args:
            image: Input image (H, W, 3)
            return_confidence: Whether to return confidence map
            
        Returns:
            InferenceResult object
        """
        # Get memory usage before inference
        memory_before = self.get_memory_usage()
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Inference with timing
        start_time = time.time()
        prediction = self.model(input_tensor, training=False)
        inference_time = time.time() - start_time
        
        # Get memory usage after inference
        memory_after = self.get_memory_usage()
        memory_usage = {
            'before_mb': memory_before['rss_mb'],
            'after_mb': memory_after['rss_mb'],
            'delta_mb': memory_after['rss_mb'] - memory_before['rss_mb']
        }
        
        # Postprocess
        class_pred, confidence = self.postprocess_prediction(prediction)
        
        return InferenceResult(
            prediction=class_pred,
            inference_time=inference_time,
            memory_usage=memory_usage,
            confidence_map=confidence if return_confidence else None
        )
    
    def batch_inference(self, images: List[np.ndarray]) -> List[InferenceResult]:
        """
        Perform batch inference on multiple images.
        
        Args:
            images: List of input images
            
        Returns:
            List of InferenceResult objects
        """
        results = []
        
        for i, image in enumerate(images):
            result = self.single_image_inference(image)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(images)} images")
        
        return results
    
    def benchmark_inference_speed(self, image: np.ndarray, 
                                num_iterations: int = 100,
                                num_warmup: int = 10) -> Dict[str, float]:
        """
        Benchmark inference speed on a single image.
        
        Args:
            image: Input image for benchmarking
            num_iterations: Number of inference iterations
            num_warmup: Number of warmup iterations
            
        Returns:
            Benchmark statistics
        """
        print(f"Benchmarking {self.model_name} inference speed...")
        
        input_tensor = self.preprocess_image(image)
        
        # Warmup
        print(f"Warmup: {num_warmup} iterations")
        for _ in range(num_warmup):
            _ = self.model(input_tensor, training=False)
        
        # Actual benchmark
        print(f"Benchmark: {num_iterations} iterations")
        times = []
        
        for i in range(num_iterations):
            start_time = time.time()
            _ = self.model(input_tensor, training=False)
            inference_time = time.time() - start_time
            times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{num_iterations}")
        
        times = np.array(times)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'p95_time': np.percentile(times, 95),
            'p99_time': np.percentile(times, 99),
            'fps': 1.0 / np.mean(times),
            'total_iterations': num_iterations
        }
    
    def visualize_prediction(self, image: np.ndarray, 
                           prediction: np.ndarray,
                           confidence: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize segmentation prediction.
        
        Args:
            image: Original image
            prediction: Class prediction map
            confidence: Confidence map (optional)
            save_path: Path to save visualization
            
        Returns:
            Visualization image
        """
        # Resize prediction to match original image size
        if image.shape[:2] != prediction.shape:
            prediction_resized = cv2.resize(
                prediction.astype(np.uint8), 
                (image.shape[1], image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        else:
            prediction_resized = prediction
        
        # Create colored segmentation
        colored_pred = self.color_lookup_bgr[prediction_resized]
        
        # Create overlay
        alpha = 0.6
        overlay = cv2.addWeighted(image, 1 - alpha, colored_pred, alpha, 0)
        
        if confidence is not None:
            # Create confidence visualization
            if confidence.shape != image.shape[:2]:
                confidence = cv2.resize(confidence, (image.shape[1], image.shape[0]))
            
            # Normalize confidence to 0-255
            conf_vis = (confidence * 255).astype(np.uint8)
            conf_colored = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
            
            # Create final visualization with confidence
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(cv2.cvtColor(colored_pred, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('Segmentation')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('Overlay')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(confidence, cmap='jet')
            axes[1, 1].set_title('Confidence Map')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
        
        if save_path and confidence is None:
            cv2.imwrite(save_path, overlay)
        
        return overlay


class RealTimeInference:
    """
    Real-time inference system for webcam/video processing.
    """
    
    def __init__(self, model_inference: ModelInference):
        self.model_inference = model_inference
        self.fps_history = []
        self.running = False
    
    def process_webcam(self, camera_id: int = 0, 
                      max_fps: Optional[float] = None,
                      display_stats: bool = True) -> None:
        """
        Process webcam feed in real-time.
        
        Args:
            camera_id: Camera device ID
            max_fps: Maximum FPS limit
            display_stats: Whether to display performance stats
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        self.running = True
        frame_count = 0
        total_inference_time = 0
        
        print("Press 'q' to quit, 's' to save current frame")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform inference
            result = self.model_inference.single_image_inference(frame, return_confidence=False)
            
            # Create visualization
            overlay = self.model_inference.visualize_prediction(frame, result.prediction)
            
            # Calculate FPS
            current_fps = 1.0 / result.inference_time if result.inference_time > 0 else 0
            self.fps_history.append(current_fps)
            
            # Keep only last 30 frames for FPS calculation
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            avg_fps = np.mean(self.fps_history)
            
            # Add performance info to display
            if display_stats:
                info_text = [
                    f"FPS: {avg_fps:.1f}",
                    f"Inference: {result.inference_time*1000:.1f}ms",
                    f"Memory: {result.memory_usage['after_mb']:.1f}MB"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(overlay, text, (10, 30 + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow(f'{self.model_inference.model_name} - Real-time Inference', overlay)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                save_path = f"realtime_capture_{timestamp}.jpg"
                cv2.imwrite(save_path, overlay)
                print(f"Saved frame to {save_path}")
            
            frame_count += 1
            total_inference_time += result.inference_time
            
            # Limit FPS if specified
            if max_fps:
                time.sleep(max(0, 1.0 / max_fps - result.inference_time))
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        avg_inference_time = total_inference_time / frame_count if frame_count > 0 else 0
        print(f"\nReal-time inference statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Average inference time: {avg_inference_time*1000:.2f}ms")
        print(f"Average FPS: {1.0/avg_inference_time:.2f}" if avg_inference_time > 0 else "N/A")
    
    def stop(self):
        """Stop real-time processing."""
        self.running = False


def benchmark_multiple_models(model_manager: ModelManager, 
                            test_image: np.ndarray,
                            model_names: List[str],
                            num_iterations: int = 100) -> Dict[str, Dict[str, float]]:
    """
    Benchmark multiple models on the same test image.
    
    Args:
        model_manager: ModelManager instance
        test_image: Test image for benchmarking
        model_names: List of model names to benchmark
        num_iterations: Number of iterations per model
        
    Returns:
        Dictionary with benchmark results for each model
    """
    results = {}
    
    for model_name in model_names:
        try:
            print(f"\nBenchmarking {model_name}...")
            model = model_manager.load_model(model_name)
            inference_system = ModelInference(model, model_name)
            
            benchmark_results = inference_system.benchmark_inference_speed(
                test_image, num_iterations=num_iterations
            )
            
            results[model_name] = benchmark_results
            
            print(f"Results for {model_name}:")
            print(f"  Mean time: {benchmark_results['mean_time']*1000:.2f}ms")
            print(f"  FPS: {benchmark_results['fps']:.2f}")
            
        except Exception as e:
            print(f"Error benchmarking {model_name}: {str(e)}")
            results[model_name] = None
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Testing ModelInference system...")
    
    # This would normally use real models and data
    # For demonstration, we'll show the structure
    
    # Create dummy test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("ModelInference system ready for use!")
    print("To use:")
    print("1. Load a model with ModelManager")
    print("2. Create ModelInference instance")
    print("3. Run inference or benchmarks")
    print("4. Use RealTimeInference for webcam processing")