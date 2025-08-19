"""
Statistical analysis tools for SegLight paper evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, normaltest
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for model evaluation.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results = {}
    
    def normality_test(self, data: np.ndarray, name: str = "data") -> Dict[str, Any]:
        """
        Test for normality using D'Agostino and Pearson's test.
        
        Args:
            data: Data array to test
            name: Name for the dataset
            
        Returns:
            Dictionary with test results
        """
        statistic, p_value = normaltest(data)
        is_normal = p_value > self.significance_level
        
        result = {
            'name': name,
            'test': "D'Agostino-Pearson",
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': is_normal,
            'alpha': self.significance_level
        }
        
        return result
    
    def compare_two_models(self, model1_metrics: np.ndarray, 
                          model2_metrics: np.ndarray,
                          model1_name: str = "Model 1",
                          model2_name: str = "Model 2",
                          metric_name: str = "metric") -> Dict[str, Any]:
        """
        Statistical comparison between two models.
        
        Args:
            model1_metrics: Metrics array for model 1
            model2_metrics: Metrics array for model 2
            model1_name: Name of first model
            model2_name: Name of second model
            metric_name: Name of the metric being compared
            
        Returns:
            Dictionary with comparison results
        """
        # Basic statistics
        stats1 = {
            'mean': np.mean(model1_metrics),
            'std': np.std(model1_metrics),
            'median': np.median(model1_metrics),
            'min': np.min(model1_metrics),
            'max': np.max(model1_metrics),
            'n': len(model1_metrics)
        }
        
        stats2 = {
            'mean': np.mean(model2_metrics),
            'std': np.std(model2_metrics),
            'median': np.median(model2_metrics),
            'min': np.min(model2_metrics),
            'max': np.max(model2_metrics),
            'n': len(model2_metrics)
        }
        
        # Test for normality
        norm1 = self.normality_test(model1_metrics, model1_name)
        norm2 = self.normality_test(model2_metrics, model2_name)
        
        # Choose appropriate test
        both_normal = norm1['is_normal'] and norm2['is_normal']
        
        if both_normal:
            # Use t-test for normal distributions
            statistic, p_value = ttest_ind(model1_metrics, model2_metrics)
            test_used = "Independent t-test"
        else:
            # Use Mann-Whitney U test for non-normal distributions
            statistic, p_value = mannwhitneyu(model1_metrics, model2_metrics, 
                                            alternative='two-sided')
            test_used = "Mann-Whitney U test"
        
        is_significant = p_value < self.significance_level
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(model1_metrics) - 1) * stats1['std']**2 + 
                             (len(model2_metrics) - 1) * stats2['std']**2) / 
                            (len(model1_metrics) + len(model2_metrics) - 2))
        cohens_d = (stats1['mean'] - stats2['mean']) / pooled_std if pooled_std > 0 else 0
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_magnitude = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_magnitude = "small"
        elif abs(cohens_d) < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
        
        result = {
            'metric_name': metric_name,
            'model1_name': model1_name,
            'model2_name': model2_name,
            'model1_stats': stats1,
            'model2_stats': stats2,
            'normality_test_1': norm1,
            'normality_test_2': norm2,
            'statistical_test': {
                'test_used': test_used,
                'statistic': statistic,
                'p_value': p_value,
                'is_significant': is_significant,
                'alpha': self.significance_level
            },
            'effect_size': {
                'cohens_d': cohens_d,
                'magnitude': effect_magnitude
            },
            'interpretation': self._interpret_comparison(stats1['mean'], stats2['mean'], 
                                                       is_significant, model1_name, model2_name)
        }
        
        return result
    
    def _interpret_comparison(self, mean1: float, mean2: float, 
                            is_significant: bool, name1: str, name2: str) -> str:
        """Generate interpretation text for model comparison."""
        if not is_significant:
            return f"No statistically significant difference between {name1} and {name2}"
        
        if mean1 > mean2:
            return f"{name1} significantly outperforms {name2}"
        else:
            return f"{name2} significantly outperforms {name1}"
    
    def multiple_model_comparison(self, models_metrics: Dict[str, np.ndarray],
                                metric_name: str = "metric") -> Dict[str, Any]:
        """
        Compare multiple models using ANOVA or Kruskal-Wallis test.
        
        Args:
            models_metrics: Dictionary with model names as keys and metrics arrays as values
            metric_name: Name of the metric being compared
            
        Returns:
            Dictionary with comparison results
        """
        model_names = list(models_metrics.keys())
        model_data = list(models_metrics.values())
        
        # Test for normality of all groups
        normality_results = {}
        all_normal = True
        
        for name, data in models_metrics.items():
            norm_result = self.normality_test(data, name)
            normality_results[name] = norm_result
            if not norm_result['is_normal']:
                all_normal = False
        
        # Choose appropriate test
        if all_normal and len(model_names) > 2:
            # Use ANOVA for normal distributions
            statistic, p_value = stats.f_oneway(*model_data)
            test_used = "One-way ANOVA"
        else:
            # Use Kruskal-Wallis test for non-normal distributions
            statistic, p_value = stats.kruskal(*model_data)
            test_used = "Kruskal-Wallis test"
        
        is_significant = p_value < self.significance_level
        
        # Calculate descriptive statistics for each model
        descriptive_stats = {}
        for name, data in models_metrics.items():
            descriptive_stats[name] = {
                'mean': np.mean(data),
                'std': np.std(data),
                'median': np.median(data),
                'min': np.min(data),
                'max': np.max(data),
                'n': len(data)
            }
        
        # Post-hoc analysis if significant
        post_hoc_results = {}
        if is_significant and len(model_names) > 2:
            post_hoc_results = self._post_hoc_analysis(models_metrics)
        
        result = {
            'metric_name': metric_name,
            'model_names': model_names,
            'descriptive_stats': descriptive_stats,
            'normality_tests': normality_results,
            'statistical_test': {
                'test_used': test_used,
                'statistic': statistic,
                'p_value': p_value,
                'is_significant': is_significant,
                'alpha': self.significance_level
            },
            'post_hoc_analysis': post_hoc_results
        }
        
        return result
    
    def _post_hoc_analysis(self, models_metrics: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform pairwise post-hoc comparisons."""
        model_names = list(models_metrics.keys())
        pairwise_results = {}
        
        for i, name1 in enumerate(model_names):
            for j, name2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{name1}_vs_{name2}"
                result = self.compare_two_models(
                    models_metrics[name1], 
                    models_metrics[name2],
                    name1, name2
                )
                pairwise_results[comparison_key] = result
        
        return pairwise_results
    
    def confidence_interval(self, data: np.ndarray, 
                          confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for the mean.
        
        Args:
            data: Data array
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)  # Standard error of the mean
        
        # Calculate t-value for the confidence level
        alpha = 1 - confidence_level
        t_value = stats.t.ppf(1 - alpha/2, n - 1)
        
        margin_error = t_value * std_err
        
        return (mean - margin_error, mean + margin_error)
    
    def performance_stability_analysis(self, metrics_over_time: np.ndarray,
                                     model_name: str = "Model") -> Dict[str, Any]:
        """
        Analyze performance stability over time/iterations.
        
        Args:
            metrics_over_time: Array of metrics in temporal order
            model_name: Name of the model
            
        Returns:
            Dictionary with stability analysis results
        """
        # Basic statistics
        mean_performance = np.mean(metrics_over_time)
        std_performance = np.std(metrics_over_time)
        cv = std_performance / mean_performance if mean_performance != 0 else float('inf')
        
        # Trend analysis (linear regression)
        x = np.arange(len(metrics_over_time))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, metrics_over_time)
        
        # Stability interpretation
        if cv < 0.1:
            stability = "Very stable"
        elif cv < 0.2:
            stability = "Stable"
        elif cv < 0.3:
            stability = "Moderately stable"
        else:
            stability = "Unstable"
        
        # Trend interpretation
        if p_value < self.significance_level:
            if slope > 0:
                trend = "Significant improving trend"
            else:
                trend = "Significant declining trend"
        else:
            trend = "No significant trend"
        
        result = {
            'model_name': model_name,
            'mean_performance': mean_performance,
            'std_performance': std_performance,
            'coefficient_of_variation': cv,
            'stability_classification': stability,
            'trend_analysis': {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_interpretation': trend
            },
            'confidence_interval_95': self.confidence_interval(metrics_over_time, 0.95)
        }
        
        return result
    
    def generate_statistical_report(self, comparison_results: List[Dict[str, Any]]) -> str:
        """
        Generate a comprehensive statistical report.
        
        Args:
            comparison_results: List of comparison results
            
        Returns:
            Formatted statistical report as string
        """
        report = []
        report.append("STATISTICAL ANALYSIS REPORT")
        report.append("=" * 50)
        report.append("")
        
        for i, result in enumerate(comparison_results, 1):
            report.append(f"{i}. {result.get('metric_name', 'Metric')} Analysis")
            report.append("-" * 30)
            
            if 'model1_name' in result:  # Two-model comparison
                model1 = result['model1_name']
                model2 = result['model2_name']
                stats1 = result['model1_stats']
                stats2 = result['model2_stats']
                test = result['statistical_test']
                effect = result['effect_size']
                
                report.append(f"Models: {model1} vs {model2}")
                report.append(f"{model1}: Mean={stats1['mean']:.4f}, SD={stats1['std']:.4f}, N={stats1['n']}")
                report.append(f"{model2}: Mean={stats2['mean']:.4f}, SD={stats2['std']:.4f}, N={stats2['n']}")
                report.append(f"Test: {test['test_used']}")
                report.append(f"Statistic: {test['statistic']:.4f}, p-value: {test['p_value']:.6f}")
                report.append(f"Significant: {'Yes' if test['is_significant'] else 'No'} (α={test['alpha']})")
                report.append(f"Effect size (Cohen's d): {effect['cohens_d']:.4f} ({effect['magnitude']})")
                report.append(f"Interpretation: {result['interpretation']}")
                
            elif 'model_names' in result:  # Multi-model comparison
                test = result['statistical_test']
                report.append(f"Models: {', '.join(result['model_names'])}")
                report.append(f"Test: {test['test_used']}")
                report.append(f"Statistic: {test['statistic']:.4f}, p-value: {test['p_value']:.6f}")
                report.append(f"Significant: {'Yes' if test['is_significant'] else 'No'} (α={test['alpha']})")
                
                if test['is_significant'] and result['post_hoc_analysis']:
                    report.append("Post-hoc pairwise comparisons:")
                    for pair, pair_result in result['post_hoc_analysis'].items():
                        pair_test = pair_result['statistical_test']
                        report.append(f"  {pair}: p={pair_test['p_value']:.6f} {'*' if pair_test['is_significant'] else ''}")
            
            report.append("")
        
        report.append("Note: * indicates statistical significance")
        
        return "\n".join(report)
    
    def save_analysis_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save analysis results to JSON file."""
        import json
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating, np.complexfloating)):
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=convert_numpy)


class BenchmarkStatistics:
    """
    Specialized statistics for benchmark comparison.
    """
    
    def __init__(self):
        self.analyzer = StatisticalAnalyzer()
    
    def compare_inference_times(self, models_times: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Compare inference times across models.
        
        Args:
            models_times: Dictionary with model names and their inference times
            
        Returns:
            Comparison results
        """
        # Convert to numpy arrays
        models_arrays = {name: np.array(times) for name, times in models_times.items()}
        
        return self.analyzer.multiple_model_comparison(models_arrays, "Inference Time")
    
    def compare_accuracy_metrics(self, models_accuracy: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Compare accuracy metrics across models.
        
        Args:
            models_accuracy: Dictionary with model names and their accuracy values
            
        Returns:
            Comparison results
        """
        # Convert to numpy arrays
        models_arrays = {name: np.array(accuracies) for name, accuracies in models_accuracy.items()}
        
        return self.analyzer.multiple_model_comparison(models_arrays, "Accuracy")
    
    def efficiency_analysis(self, models_data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Analyze efficiency (accuracy per unit time).
        
        Args:
            models_data: Dictionary with model data including 'accuracy' and 'inference_time'
            
        Returns:
            DataFrame with efficiency analysis
        """
        efficiency_data = []
        
        for model_name, data in models_data.items():
            accuracy = data.get('accuracy', 0)
            time = data.get('inference_time', 1)  # Avoid division by zero
            
            efficiency = accuracy / time if time > 0 else 0
            
            efficiency_data.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Inference_Time': time,
                'Efficiency': efficiency,
                'FPS': 1.0 / time if time > 0 else 0
            })
        
        return pd.DataFrame(efficiency_data).sort_values('Efficiency', ascending=False)


if __name__ == "__main__":
    # Example usage
    analyzer = StatisticalAnalyzer()
    
    # Generate sample data for demonstration
    np.random.seed(42)
    
    model1_iou = np.random.normal(0.85, 0.05, 100)  # Model 1 IoU scores
    model2_iou = np.random.normal(0.82, 0.04, 100)  # Model 2 IoU scores
    model3_iou = np.random.normal(0.88, 0.06, 100)  # Model 3 IoU scores
    
    # Two-model comparison
    comparison = analyzer.compare_two_models(
        model1_iou, model2_iou, 
        "MazeNet", "Baseline", "IoU"
    )
    
    print("Two-model comparison:")
    print(f"MazeNet mean IoU: {comparison['model1_stats']['mean']:.4f}")
    print(f"Baseline mean IoU: {comparison['model2_stats']['mean']:.4f}")
    print(f"Test used: {comparison['statistical_test']['test_used']}")
    print(f"p-value: {comparison['statistical_test']['p_value']:.6f}")
    print(f"Significant: {comparison['statistical_test']['is_significant']}")
    print(f"Effect size: {comparison['effect_size']['cohens_d']:.4f} ({comparison['effect_size']['magnitude']})")
    print()
    
    # Multi-model comparison
    models_data = {
        "MazeNet": model1_iou,
        "Baseline": model2_iou,
        "Enhanced": model3_iou
    }
    
    multi_comparison = analyzer.multiple_model_comparison(models_data, "IoU")
    
    print("Multi-model comparison:")
    print(f"Test used: {multi_comparison['statistical_test']['test_used']}")
    print(f"p-value: {multi_comparison['statistical_test']['p_value']:.6f}")
    print(f"Significant: {multi_comparison['statistical_test']['is_significant']}")
    
    # Generate report
    report = analyzer.generate_statistical_report([comparison, multi_comparison])
    print("\nStatistical Report:")
    print(report)