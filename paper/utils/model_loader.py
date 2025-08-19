"""
Model loading and management utilities for SegLight paper evaluation.
"""

import os
import sys
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import *
from network import MazeNet


class ModelManager:
    """
    Manages multiple models for comparison and evaluation.
    """
    
    def __init__(self, models_dir: str = "paper/models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.model_info = {}
        
    def register_model(self, name: str, model_path: str, model_type: str = "tensorflow", 
                      description: str = "", **kwargs):
        """
        Register a model for evaluation.
        
        Args:
            name: Unique identifier for the model
            model_path: Path to the model file/directory
            model_type: Type of model ('tensorflow', 'keras', 'custom')
            description: Description of the model
            **kwargs: Additional model metadata
        """
        self.model_info[name] = {
            'path': model_path,
            'type': model_type,
            'description': description,
            'metadata': kwargs
        }
        
    def load_model(self, name: str, compile_model: bool = False):
        """
        Load a registered model.
        
        Args:
            name: Model name
            compile_model: Whether to compile the model
            
        Returns:
            Loaded model instance
        """
        if name in self.loaded_models:
            return self.loaded_models[name]
            
        if name not in self.model_info:
            raise ValueError(f"Model '{name}' not registered")
            
        info = self.model_info[name]
        model_path = info['path']
        model_type = info['type']
        
        try:
            if model_type == 'tensorflow' or model_type == 'keras':
                if os.path.exists(model_path):
                    model = tf.keras.models.load_model(model_path, compile=compile_model)
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
                    
            elif model_type == 'custom':
                # For custom models like MazeNet
                if name.lower() == 'mazenet' or 'maze' in name.lower():
                    model = MazeNet()
                    if os.path.exists(model_path):
                        # Load weights if available
                        model.load_weights(model_path)
                else:
                    raise ValueError(f"Unknown custom model type: {name}")
                    
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            self.loaded_models[name] = model
            print(f"Successfully loaded model: {name}")
            return model
            
        except Exception as e:
            print(f"Error loading model {name}: {str(e)}")
            raise
            
    def get_model_info(self, name: str) -> Dict:
        """Get model information."""
        if name not in self.model_info:
            raise ValueError(f"Model '{name}' not registered")
        return self.model_info[name].copy()
        
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.model_info.keys())
        
    def get_model_summary(self, name: str) -> Dict[str, Any]:
        """
        Get detailed model summary including parameters count, size, etc.
        
        Args:
            name: Model name
            
        Returns:
            Dictionary with model summary information
        """
        model = self.load_model(name)
        
        # Calculate model parameters
        total_params = model.count_params() if hasattr(model, 'count_params') else 0
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        # Estimate model size (in MB)
        model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Get model input/output shapes
        input_shape = None
        output_shape = None
        try:
            if hasattr(model, 'input_shape'):
                input_shape = model.input_shape
            if hasattr(model, 'output_shape'):
                output_shape = model.output_shape
        except:
            pass
            
        summary = {
            'name': name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'model_size_mb': model_size_mb,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'info': self.get_model_info(name)
        }
        
        return summary
        
    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple models in terms of size, parameters, etc.
        
        Args:
            model_names: List of model names to compare
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'models': [],
            'summary': {}
        }
        
        for name in model_names:
            try:
                summary = self.get_model_summary(name)
                comparison['models'].append(summary)
            except Exception as e:
                print(f"Error getting summary for {name}: {str(e)}")
                
        # Calculate comparison statistics
        if comparison['models']:
            param_counts = [m['total_parameters'] for m in comparison['models']]
            sizes = [m['model_size_mb'] for m in comparison['models']]
            
            comparison['summary'] = {
                'total_models': len(comparison['models']),
                'parameter_range': {'min': min(param_counts), 'max': max(param_counts)},
                'size_range_mb': {'min': min(sizes), 'max': max(sizes)},
                'lightest_model': min(comparison['models'], key=lambda x: x['total_parameters'])['name'],
                'largest_model': max(comparison['models'], key=lambda x: x['total_parameters'])['name']
            }
            
        return comparison


def create_default_models() -> ModelManager:
    """
    Create a ModelManager with default models for evaluation.
    
    Returns:
        Configured ModelManager instance
    """
    manager = ModelManager()
    
    # Register MazeNet (the main model)
    manager.register_model(
        name="MazeNet",
        model_path="",  # Will be created fresh
        model_type="custom",
        description="Lightweight semantic segmentation model for soccer field",
        architecture="Custom CNN with separable convolutions and SPP",
        target_application="Humanoid soccer robots",
        input_size=(INPUT_HEIGHT, INPUT_WIDTH, 3),
        output_classes=NUMBER_OF_CLASSES
    )
    
    # Add pre-trained model if exists
    if os.path.exists("models/best_model"):
        manager.register_model(
            name="MazeNet_Pretrained",
            model_path="models/best_model",
            model_type="tensorflow",
            description="Pre-trained MazeNet model",
            architecture="MazeNet",
            training_epochs="Best model from training"
        )
    
    return manager


def load_test_data(batch_size: int = 32) -> tf.data.Dataset:
    """
    Load test dataset for evaluation.
    
    Args:
        batch_size: Batch size for the dataset
        
    Returns:
        Test dataset
    """
    from data_provider import getData
    _, test_set = getData()
    return test_set.batch(batch_size) if batch_size else test_set


if __name__ == "__main__":
    # Example usage
    manager = create_default_models()
    
    print("Available models:")
    for model_name in manager.list_models():
        print(f"- {model_name}")
        
    # Load and summarize MazeNet
    try:
        model = manager.load_model("MazeNet")
        summary = manager.get_model_summary("MazeNet")
        print(f"\nMazeNet Summary:")
        print(f"Total parameters: {summary['total_parameters']:,}")
        print(f"Model size: {summary['model_size_mb']:.2f} MB")
    except Exception as e:
        print(f"Error: {e}")