"""
Comprehensive visualization and plotting system for SegLight paper.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import cv2
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import COLOR_MAP, NUMBER_OF_CLASSES


class PaperVisualizer:
    """
    Publication-ready visualization system for SegLight paper.
    """
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (10, 6)):
        self.style = style
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 8)
        self.class_colors = self._setup_class_colors()
        
        # Set publication style
        plt.style.use('default')  # Reset to default first
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'figure.figsize': self.figsize,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.transparent': False,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def _setup_class_colors(self) -> Dict[str, str]:
        """Setup consistent colors for classes."""
        colors = {}
        class_names = list(COLOR_MAP.keys())
        
        for i, class_name in enumerate(class_names):
            colors[class_name] = self.color_palette[i % len(self.color_palette)]
        
        return colors
    
    def plot_training_curves(self, training_logs: Dict[str, List[float]], 
                           save_path: Optional[str] = None) -> None:
        """
        Plot training curves (loss and IoU over epochs).
        
        Args:
            training_logs: Dictionary with 'epochs', 'train_loss', 'val_loss', 'train_iou', 'val_iou'
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = training_logs.get('epochs', range(len(training_logs.get('train_loss', []))))
        
        # Plot loss curves
        if 'train_loss' in training_logs:
            ax1.plot(epochs, training_logs['train_loss'], 
                    label='Training Loss', color=self.color_palette[0], linewidth=2)
        if 'val_loss' in training_logs:
            ax1.plot(epochs, training_logs['val_loss'], 
                    label='Validation Loss', color=self.color_palette[1], linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot IoU curves
        if 'train_iou' in training_logs:
            ax2.plot(epochs, training_logs['train_iou'], 
                    label='Training IoU', color=self.color_palette[2], linewidth=2)
        if 'val_iou' in training_logs:
            ax2.plot(epochs, training_logs['val_iou'], 
                    label='Validation IoU', color=self.color_palette[3], linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean IoU')
        ax2.set_title('Training and Validation IoU')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, comparison_data: pd.DataFrame, 
                            metrics: List[str] = ['FPS', 'Mean IoU', 'Pixel Accuracy'],
                            save_path: Optional[str] = None) -> None:
        """
        Plot model comparison across multiple metrics.
        
        Args:
            comparison_data: DataFrame with models and their metrics
            metrics: List of metrics to compare
            save_path: Path to save the plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in comparison_data.columns:
                bars = axes[i].bar(comparison_data['Model'], comparison_data[metric], 
                                 color=self.color_palette[:len(comparison_data)],
                                 alpha=0.8)
                
                axes[i].set_title(f'{metric} Comparison')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Model comparison saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, 
                            class_names: List[str] = None,
                            normalize: bool = True,
                            save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
        """
        if class_names is None:
            class_names = list(COLOR_MAP.keys())
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Frequency' if not normalize else 'Proportion'})
        
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_performance_vs_accuracy(self, models_data: List[Dict[str, float]],
                                   save_path: Optional[str] = None) -> None:
        """
        Plot performance (FPS) vs accuracy trade-off.
        
        Args:
            models_data: List of dictionaries with model metrics
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for i, data in enumerate(models_data):
            plt.scatter(data.get('fps', 0), data.get('mean_iou', 0),
                       s=200, alpha=0.7, color=self.color_palette[i % len(self.color_palette)],
                       label=data.get('model_name', f'Model {i+1}'))
            
            # Add model name annotations
            plt.annotate(data.get('model_name', f'Model {i+1}'),
                        (data.get('fps', 0), data.get('mean_iou', 0)),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, alpha=0.8)
        
        plt.xlabel('Frames Per Second (FPS)')
        plt.ylabel('Mean IoU')
        plt.title('Performance vs Accuracy Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Performance vs accuracy plot saved to {save_path}")
        
        plt.show()
    
    def plot_class_performance(self, class_metrics: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = None) -> None:
        """
        Plot per-class performance metrics.
        
        Args:
            class_metrics: Dictionary with class names and their metrics
            save_path: Path to save the plot
        """
        # Prepare data
        classes = list(class_metrics.keys())
        metrics = ['IoU', 'Accuracy', 'Precision', 'Recall'] if class_metrics else []
        
        # Filter metrics that exist in the data
        available_metrics = []
        for metric in metrics:
            if any(metric.lower() in class_metrics[cls] for cls in classes):
                available_metrics.append(metric)
        
        if not available_metrics:
            # Use the actual metrics available
            if classes:
                available_metrics = list(class_metrics[classes[0]].keys())
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            values = []
            for cls in classes:
                # Try different key formats
                value = (class_metrics[cls].get(metric) or 
                        class_metrics[cls].get(metric.lower()) or 
                        class_metrics[cls].get(f"{metric.lower()}_per_class", 0))
                values.append(value)
            
            bars = axes[i].bar(classes, values, 
                             color=[self.class_colors.get(cls, self.color_palette[j % len(self.color_palette)]) 
                                   for j, cls in enumerate(classes)],
                             alpha=0.8)
            
            axes[i].set_title(f'{metric} per Class')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height is not None:
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}',
                               ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Class performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_inference_time_distribution(self, inference_times: List[float],
                                       model_name: str = "Model",
                                       save_path: Optional[str] = None) -> None:
        """
        Plot distribution of inference times.
        
        Args:
            inference_times: List of inference times in seconds
            model_name: Name of the model
            save_path: Path to save the plot
        """
        times_ms = np.array(inference_times) * 1000  # Convert to milliseconds
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(times_ms, bins=30, alpha=0.7, color=self.color_palette[0], edgecolor='black')
        ax1.axvline(np.mean(times_ms), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(times_ms):.2f}ms')
        ax1.axvline(np.median(times_ms), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(times_ms):.2f}ms')
        ax1.set_xlabel('Inference Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{model_name} - Inference Time Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(times_ms, vert=True, patch_artist=True,
                   boxprops=dict(facecolor=self.color_palette[1], alpha=0.7))
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title(f'{model_name} - Inference Time Statistics')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Statistics:
Mean: {np.mean(times_ms):.2f}ms
Std: {np.std(times_ms):.2f}ms
Min: {np.min(times_ms):.2f}ms
Max: {np.max(times_ms):.2f}ms
P95: {np.percentile(times_ms, 95):.2f}ms
FPS: {1000/np.mean(times_ms):.2f}"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Inference time distribution saved to {save_path}")
        
        plt.show()
    
    def create_segmentation_showcase(self, images: List[np.ndarray],
                                   predictions: List[np.ndarray],
                                   ground_truths: List[np.ndarray],
                                   titles: List[str] = None,
                                   save_path: Optional[str] = None) -> None:
        """
        Create a showcase of segmentation results.
        
        Args:
            images: List of original images
            predictions: List of prediction masks
            ground_truths: List of ground truth masks
            titles: List of titles for each image
            save_path: Path to save the showcase
        """
        n_images = len(images)
        fig, axes = plt.subplots(3, n_images, figsize=(4 * n_images, 12))
        
        if n_images == 1:
            axes = axes.reshape(-1, 1)
        
        # Color mapping for visualization
        color_lookup_bgr = np.zeros((len(COLOR_MAP), 3), dtype=np.uint8)
        for idx, (class_name, color) in enumerate(COLOR_MAP.items()):
            color_lookup_bgr[idx] = np.array(color, dtype=np.uint8)
        
        for i in range(n_images):
            # Original image
            axes[0, i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(titles[i] if titles else f'Image {i+1}')
            axes[0, i].axis('off')
            
            # Ground truth
            gt_colored = color_lookup_bgr[ground_truths[i]]
            axes[1, i].imshow(gt_colored)
            axes[1, i].set_title('Ground Truth')
            axes[1, i].axis('off')
            
            # Prediction
            pred_colored = color_lookup_bgr[predictions[i]]
            axes[2, i].imshow(pred_colored)
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')
        
        # Add legend
        legend_elements = []
        for class_name, color in COLOR_MAP.items():
            legend_elements.append(mpatches.Patch(color=np.array(color)/255.0, label=class_name))
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=len(COLOR_MAP))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Segmentation showcase saved to {save_path}")
        
        plt.show()
    
    def create_architecture_diagram(self, save_path: Optional[str] = None) -> None:
        """
        Create a simplified architecture diagram for MazeNet.
        
        Args:
            save_path: Path to save the diagram
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Define layer positions and sizes
        layers = [
            {"name": "Input\n(320×240×3)", "pos": (1, 4), "size": (1.5, 1), "color": "lightblue"},
            {"name": "Conv2D\n(6 filters)", "pos": (3, 4), "size": (1.5, 1), "color": "lightcoral"},
            {"name": "SepConv2D\n(12 filters)", "pos": (5, 4), "size": (1.5, 1), "color": "lightgreen"},
            {"name": "AvgPool2D", "pos": (7, 4), "size": (1, 0.8), "color": "lightyellow"},
            {"name": "SepConv2D\n(12 filters)", "pos": (9, 4), "size": (1.5, 1), "color": "lightgreen"},
            {"name": "AvgPool2D", "pos": (11, 4), "size": (1, 0.8), "color": "lightyellow"},
            {"name": "SPP\nModule", "pos": (13, 4), "size": (2, 2), "color": "lightpink"},
            {"name": "Feature\nFusion", "pos": (9, 2), "size": (2, 1), "color": "lightcyan"},
            {"name": "SepConv2D\n(3 classes)", "pos": (7, 2), "size": (1.5, 1), "color": "lightgreen"},
            {"name": "Softmax\nOutput", "pos": (5, 2), "size": (1.5, 1), "color": "lightsteelblue"},
        ]
        
        # Draw layers
        for layer in layers:
            rect = Rectangle(layer["pos"], layer["size"][0], layer["size"][1], 
                           facecolor=layer["color"], edgecolor="black", linewidth=1.5)
            ax.add_patch(rect)
            
            # Add text
            text_x = layer["pos"][0] + layer["size"][0] / 2
            text_y = layer["pos"][1] + layer["size"][1] / 2
            ax.text(text_x, text_y, layer["name"], ha="center", va="center", 
                   fontsize=10, fontweight="bold")
        
        # Draw connections (arrows)
        connections = [
            ((2.5, 4.5), (3, 4.5)),  # Input to Conv2D
            ((4.5, 4.5), (5, 4.5)),  # Conv2D to SepConv2D
            ((6.5, 4.5), (7, 4.4)),  # SepConv2D to AvgPool2D
            ((8, 4.4), (9, 4.5)),    # AvgPool2D to SepConv2D
            ((10.5, 4.5), (11, 4.4)), # SepConv2D to AvgPool2D
            ((12, 4.4), (13, 4.5)),  # AvgPool2D to SPP
            ((13.5, 4), (10, 2.5)),  # SPP to Feature Fusion
            ((4.5, 4), (9.5, 3)),    # Conv2D to Feature Fusion (skip connection)
            ((9, 2.5), (8.5, 2.5)),  # Feature Fusion to SepConv2D
            ((7, 2.5), (6.5, 2.5)),  # SepConv2D to Softmax
        ]
        
        for start, end in connections:
            ax.annotate("", xy=end, xytext=start,
                       arrowprops=dict(arrowstyle="->", lw=1.5, color="darkblue"))
        
        ax.set_xlim(0, 16)
        ax.set_ylim(1, 6)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('MazeNet Architecture', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Architecture diagram saved to {save_path}")
        
        plt.show()


def create_publication_plots(results_data: Dict[str, Any], 
                           output_dir: str = "paper/visualization/plots") -> None:
    """
    Create all publication-ready plots for the paper.
    
    Args:
        results_data: Dictionary containing all experimental results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    visualizer = PaperVisualizer()
    
    print("Creating publication plots...")
    
    # 1. Training curves
    if 'training_logs' in results_data:
        visualizer.plot_training_curves(
            results_data['training_logs'],
            save_path=os.path.join(output_dir, 'training_curves.png')
        )
    
    # 2. Model comparison
    if 'model_comparison' in results_data:
        visualizer.plot_model_comparison(
            results_data['model_comparison'],
            save_path=os.path.join(output_dir, 'model_comparison.png')
        )
    
    # 3. Confusion matrix
    if 'confusion_matrix' in results_data:
        visualizer.plot_confusion_matrix(
            results_data['confusion_matrix'],
            save_path=os.path.join(output_dir, 'confusion_matrix.png')
        )
    
    # 4. Performance vs accuracy
    if 'performance_accuracy' in results_data:
        visualizer.plot_performance_vs_accuracy(
            results_data['performance_accuracy'],
            save_path=os.path.join(output_dir, 'performance_vs_accuracy.png')
        )
    
    # 5. Architecture diagram
    visualizer.create_architecture_diagram(
        save_path=os.path.join(output_dir, 'mazenet_architecture.png')
    )
    
    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    visualizer = PaperVisualizer()
    
    # Create sample data for demonstration
    sample_training_logs = {
        'epochs': list(range(1, 51)),
        'train_loss': [0.8 - 0.01*i + 0.05*np.random.random() for i in range(50)],
        'val_loss': [0.85 - 0.008*i + 0.08*np.random.random() for i in range(50)],
        'train_iou': [0.3 + 0.01*i + 0.02*np.random.random() for i in range(50)],
        'val_iou': [0.25 + 0.008*i + 0.03*np.random.random() for i in range(50)]
    }
    
    visualizer.plot_training_curves(sample_training_logs)
    print("Sample visualization created!")