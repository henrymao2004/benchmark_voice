#!/usr/bin/env python3
"""
Detector Overall Score Calculation

This module implements the proposed Detector Overall Score (S_ovrl) from the paper:
"SoK: Benchmarking Fake Voice Detection in the Fake Voice Generation Arms Race"

The score combines empirical performance against multiple fake voice generators 
with a penalty for model complexity, providing a unified metric for detector evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectorOverallScore:
    """
    Calculate the overall detector score that combines empirical performance 
    with model complexity penalty.
    
    The score is computed as:
    S_i = α * P_i + (1-α) * Ĉ_i
    
    Where:
    - P_i: Empirical performance across all generators (weighted average)
    - Ĉ_i: Normalized log-parameter count (model complexity penalty)
    - α: Trade-off factor between performance and efficiency
    """
    
    def __init__(self, alpha: float = 0.8):
        """
        Initialize the detector overall score calculator.
        
        Args:
            alpha: Trade-off factor between empirical performance and model complexity.
                  Higher values emphasize performance over efficiency.
                  Range: [0, 1], default: 0.8
        """
        self.alpha = alpha
        self.generator_challenge_scores = {}
        self.detector_parameters = {}
        
    def set_generator_challenge_scores(self, challenge_scores: Dict[str, float]):
        """
        Set the challenge scores for each generator.
        
        Args:
            challenge_scores: Dictionary mapping generator names to challenge scores.
                            Higher scores indicate more challenging generators.
        """
        self.generator_challenge_scores = challenge_scores
        logger.info(f"Set challenge scores for {len(challenge_scores)} generators")
        
    def set_detector_parameters(self, parameters: Dict[str, int]):
        """
        Set the parameter counts for each detector.
        
        Args:
            parameters: Dictionary mapping detector names to parameter counts (in millions).
        """
        self.detector_parameters = parameters
        logger.info(f"Set parameters for {len(parameters)} detectors")
        
    def normalize_metrics(self, metrics: Dict[str, Dict[str, float]], metric_type: str) -> Dict[str, Dict[str, float]]:
        """
        Normalize metrics to [0,1] range where lower values indicate better performance.
        
        Args:
            metrics: Nested dictionary {detector: {generator: metric_value}}
            metric_type: Type of metric ('EER', 'C_llr', 'minDCF')
            
        Returns:
            Normalized metrics in the same structure
        """
        # Flatten all values to find global min/max
        all_values = []
        for detector_metrics in metrics.values():
            all_values.extend(detector_metrics.values())
        
        if not all_values:
            return metrics
            
        min_val = min(all_values)
        max_val = max(all_values)
        
        normalized_metrics = {}
        for detector, generator_metrics in metrics.items():
            normalized_metrics[detector] = {}
            for generator, value in generator_metrics.items():
                if max_val != min_val:
                    normalized_value = (value - min_val) / (max_val - min_val)
                else:
                    normalized_value = 0.0
                normalized_metrics[detector][generator] = normalized_value
                
        logger.info(f"Normalized {metric_type} metrics: min={min_val:.4f}, max={max_val:.4f}")
        return normalized_metrics
    
    def calculate_generator_weights(self) -> Dict[str, float]:
        """
        Calculate normalized weights for each generator based on challenge scores.
        
        Returns:
            Dictionary mapping generator names to normalized weights
        """
        if not self.generator_challenge_scores:
            # If no challenge scores provided, use uniform weights
            generators = list(self.generator_challenge_scores.keys())
            return {gen: 1.0/len(generators) for gen in generators}
            
        total_score = sum(self.generator_challenge_scores.values())
        weights = {gen: score/total_score for gen, score in self.generator_challenge_scores.items()}
        
        logger.info(f"Generator weights calculated (total challenge score: {total_score})")
        return weights
    
    def calculate_empirical_performance(self, 
                                      eer_metrics: Dict[str, Dict[str, float]],
                                      cllr_metrics: Dict[str, Dict[str, float]], 
                                      mindcf_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate empirical performance P_i for each detector.
        
        Args:
            eer_metrics: EER values for each detector-generator pair
            cllr_metrics: C_llr values for each detector-generator pair  
            mindcf_metrics: minDCF values for each detector-generator pair
            
        Returns:
            Dictionary mapping detector names to empirical performance scores
        """
        # Normalize all metrics
        norm_eer = self.normalize_metrics(eer_metrics, 'EER')
        norm_cllr = self.normalize_metrics(cllr_metrics, 'C_llr')
        norm_mindcf = self.normalize_metrics(mindcf_metrics, 'minDCF')
        
        # Calculate generator weights
        generator_weights = self.calculate_generator_weights()
        
        empirical_scores = {}
        
        for detector in eer_metrics.keys():
            weighted_sum = 0.0
            
            for generator in eer_metrics[detector].keys():
                if generator in generator_weights:
                    # Calculate average of three normalized metrics for this detector-generator pair
                    avg_metric = (norm_eer[detector][generator] + 
                                 norm_cllr[detector][generator] + 
                                 norm_mindcf[detector][generator]) / 3.0
                    
                    # Weight by generator challenge score
                    weighted_sum += generator_weights[generator] * avg_metric
            
            empirical_scores[detector] = weighted_sum
            
        logger.info(f"Calculated empirical performance for {len(empirical_scores)} detectors")
        return empirical_scores
    
    def calculate_complexity_penalty(self) -> Dict[str, float]:
        """
        Calculate normalized log-parameter count penalty for each detector.
        
        Returns:
            Dictionary mapping detector names to complexity penalty scores
        """
        if not self.detector_parameters:
            logger.warning("No detector parameters provided, using uniform complexity penalty")
            return {}
            
        # Log-transform parameter counts
        log_params = {detector: np.log(params) for detector, params in self.detector_parameters.items()}
        
        min_log = min(log_params.values())
        max_log = max(log_params.values())
        
        complexity_penalties = {}
        for detector, log_param in log_params.items():
            if max_log != min_log:
                normalized_penalty = (log_param - min_log) / (max_log - min_log)
            else:
                normalized_penalty = 0.0
            complexity_penalties[detector] = normalized_penalty
            
        logger.info(f"Complexity penalties calculated: min_log_params={min_log:.2f}, max_log_params={max_log:.2f}")
        return complexity_penalties
    
    def calculate_overall_scores(self,
                               eer_metrics: Dict[str, Dict[str, float]],
                               cllr_metrics: Dict[str, Dict[str, float]], 
                               mindcf_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate the overall detector scores S_i.
        
        Args:
            eer_metrics: EER values for each detector-generator pair
            cllr_metrics: C_llr values for each detector-generator pair
            mindcf_metrics: minDCF values for each detector-generator pair
            
        Returns:
            Dictionary mapping detector names to overall scores (lower is better)
        """
        # Calculate empirical performance
        empirical_performance = self.calculate_empirical_performance(eer_metrics, cllr_metrics, mindcf_metrics)
        
        # Calculate complexity penalty
        complexity_penalty = self.calculate_complexity_penalty()
        
        # Combine using the trade-off factor
        overall_scores = {}
        for detector in empirical_performance.keys():
            emp_perf = empirical_performance[detector]
            comp_penalty = complexity_penalty.get(detector, 0.0)
            
            overall_score = self.alpha * emp_perf + (1 - self.alpha) * comp_penalty
            overall_scores[detector] = overall_score
            
        logger.info(f"Overall scores calculated with alpha={self.alpha}")
        return overall_scores
    
    def rank_detectors(self, overall_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Rank detectors by their overall scores (ascending order).
        
        Args:
            overall_scores: Dictionary of detector names to overall scores
            
        Returns:
            List of (detector_name, score) tuples sorted by score (best first)
        """
        ranked = sorted(overall_scores.items(), key=lambda x: x[1])
        return ranked
    
    def generate_report(self, 
                       overall_scores: Dict[str, float],
                       empirical_performance: Dict[str, float],
                       complexity_penalty: Dict[str, float]) -> str:
        """
        Generate a detailed report of the evaluation results.
        
        Args:
            overall_scores: Overall scores for each detector
            empirical_performance: Empirical performance scores
            complexity_penalty: Complexity penalty scores
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("DETECTOR OVERALL SCORE EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"Trade-off factor (α): {self.alpha}")
        report.append(f"Number of detectors evaluated: {len(overall_scores)}")
        report.append("")
        
        # Rank detectors
        ranked_detectors = self.rank_detectors(overall_scores)
        
        report.append("DETECTOR RANKINGS (lower score = better performance):")
        report.append("-" * 50)
        for rank, (detector, score) in enumerate(ranked_detectors, 1):
            emp_perf = empirical_performance.get(detector, 0.0)
            comp_penalty = complexity_penalty.get(detector, 0.0)
            params = self.detector_parameters.get(detector, 0)
            
            report.append(f"{rank:2d}. {detector:20s}")
            report.append(f"     Overall Score: {score:.4f}")
            report.append(f"     Empirical Performance: {emp_perf:.4f}")
            report.append(f"     Complexity Penalty: {comp_penalty:.4f}")
            report.append(f"     Parameters: {params:,}M")
            report.append("")
            
        return "\n".join(report)


def load_metrics_from_csv(eer_file: str, cllr_file: str, mindcf_file: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load metrics from CSV files.
    
    Args:
        eer_file: Path to EER metrics CSV file
        cllr_file: Path to C_llr metrics CSV file  
        mindcf_file: Path to minDCF metrics CSV file
        
    Returns:
        Tuple of (eer_metrics, cllr_metrics, mindcf_metrics) dictionaries
    """
    def load_metric_file(filepath: str) -> Dict[str, Dict[str, float]]:
        """Load a single metric file and convert to nested dictionary format."""
        try:
            df = pd.read_csv(filepath, index_col=0)
            return df.to_dict()
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return {}
    
    eer_metrics = load_metric_file(eer_file)
    cllr_metrics = load_metric_file(cllr_file)
    mindcf_metrics = load_metric_file(mindcf_file)
    
    return eer_metrics, cllr_metrics, mindcf_metrics


def main():
    """Example usage of the DetectorOverallScore class."""
    
    # Example datas, just for demonstration purposes
    example_eer = {
        'XLSR-SLS': {'MaskGCT-TTS': 24.94, 'FireRedTTS-1S': 30.86, 'BigVGAN': 25.92},
        'XLSR-Conformer+TCM': {'MaskGCT-TTS': 28.15, 'FireRedTTS-1S': 30.86, 'BigVGAN': 22.45},
        'RawNet2': {'MaskGCT-TTS': 45.23, 'FireRedTTS-1S': 52.18, 'BigVGAN': 38.92}
    }
    
    example_cllr = {
        'XLSR-SLS': {'MaskGCT-TTS': 0.85, 'FireRedTTS-1S': 0.92, 'BigVGAN': 0.78},
        'XLSR-Conformer+TCM': {'MaskGCT-TTS': 0.82, 'FireRedTTS-1S': 0.89, 'BigVGAN': 0.75},
        'RawNet2': {'MaskGCT-TTS': 1.23, 'FireRedTTS-1S': 1.45, 'BigVGAN': 1.12}
    }
    
    example_mindcf = {
        'XLSR-SLS': {'MaskGCT-TTS': 0.45, 'FireRedTTS-1S': 0.52, 'BigVGAN': 0.38},
        'XLSR-Conformer+TCM': {'MaskGCT-TTS': 0.42, 'FireRedTTS-1S': 0.48, 'BigVGAN': 0.35},
        'RawNet2': {'MaskGCT-TTS': 0.68, 'FireRedTTS-1S': 0.75, 'BigVGAN': 0.62}
    }
    
    # Initialize calculator
    calculator = DetectorOverallScore(alpha=0.8)
    
    # Set generator challenge scores (higher = more challenging), examples
    challenge_scores = {
        'MaskGCT-TTS': 3.0,  # Neural codec language model - most challenging
        'FireRedTTS-1S': 3.0,  # Neural codec language model - most challenging  
        'BigVGAN': 2.0,  # Advanced GAN vocoder - moderately challenging
    }
    calculator.set_generator_challenge_scores(challenge_scores)
    
    # Set detector parameters (in millions), examples
    detector_params = {
        'XLSR-SLS': 317,  # Large transformer-based model
        'XLSR-Conformer+TCM': 317,  # Large transformer-based model
        'RawNet2': 4.5,  # Lightweight end-to-end model
    }
    calculator.set_detector_parameters(detector_params)
    
    # Calculate overall scores
    overall_scores = calculator.calculate_overall_scores(example_eer, example_cllr, example_mindcf)
    empirical_performance = calculator.calculate_empirical_performance(example_eer, example_cllr, example_mindcf)
    complexity_penalty = calculator.calculate_complexity_penalty()
    
    # Generate and print report
    report = calculator.generate_report(overall_scores, empirical_performance, complexity_penalty)
    print(report)
    
    # Save results
    results_df = pd.DataFrame({
        'Detector': list(overall_scores.keys()),
        'Overall_Score': list(overall_scores.values()),
        'Empirical_Performance': [empirical_performance[d] for d in overall_scores.keys()],
        'Complexity_Penalty': [complexity_penalty[d] for d in overall_scores.keys()],
        'Parameters_M': [detector_params[d] for d in overall_scores.keys()]
    })
    
    results_df = results_df.sort_values('Overall_Score')
    results_df.to_csv('detector_overall_scores.csv', index=False)
    print(f"\nResults saved to 'detector_overall_scores.csv'")


if __name__ == "__main__":
    main() 