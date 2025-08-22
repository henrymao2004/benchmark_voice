#!/usr/bin/env python3
"""
Generator Overall Score Calculation

This module implements the proposed Generator Overall Score from the paper:
"SoK: Benchmarking Fake Voice Detection in the Fake Voice Generation Arms Race"

The score evaluates fake voice generators using weighted aggregation of normalized metrics
for TTS systems and audio reconstruction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneratorOverallScore:
    """
    Calculate the overall generator score that combines multiple evaluation metrics
    for TTS systems and audio reconstruction models.
    
    For TTS systems:
    F_TTS = Σ(α_i * M_i,norm)
    
    For Audio Reconstruction:
    F_Audio_Reconstruction = Σ(β_j * M_j,norm)
    
    Where M_i,norm and M_j,norm are normalized metrics in range (0,1].
    """
    
    def __init__(self):
        """Initialize the generator overall score calculator."""
        
        # TTS system weights (α_i)
        self.tts_weights = {
            'DNSMOS': 0.25,      # Perceptual quality
            'SIM': 0.25,         # Speaker similarity
            'PSNR': 0.15,        # Signal fidelity
            'WER': 0.20,         # Intelligibility (inverted)
            'RTF': 0.075,        # Computational efficiency (inverted)
            'LogParams': 0.075   # Model size (inverted)
        }
        
        # Audio reconstruction weights (β_j)
        self.audio_recon_weights = {
            'DNSMOS': 0.25,      # Perceptual quality
            'PESQ': 0.225,       # Perceptual quality
            'PSNR': 0.15,        # Signal fidelity
            'SSIM': 0.225,       # Frequency-domain similarity
            'RTF': 0.075,        # Computational efficiency (inverted)
            'LogParams': 0.075   # Model size (inverted)
        }
        
        # Validate weights sum to 1.0
        assert abs(sum(self.tts_weights.values()) - 1.0) < 1e-6, "TTS weights must sum to 1.0"
        assert abs(sum(self.audio_recon_weights.values()) - 1.0) < 1e-6, "Audio recon weights must sum to 1.0"
        
    def normalize_metric(self, values: List[float], metric_name: str, epsilon: float = 0.1) -> List[float]:
        """
        Normalize metric values to range (0,1].
        
        Args:
            values: List of raw metric values
            metric_name: Name of the metric for logging
            epsilon: Small constant to ensure values > 0
            
        Returns:
            List of normalized values in range (epsilon, 1]
        """
        if not values:
            return []
            
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            # All values are the same, assign uniform normalized values
            normalized = [0.5 + epsilon/2] * len(values)
        else:
            # Normalize to [0,1] then shift to (epsilon, 1]
            normalized = [(val - min_val) / (max_val - min_val) * (1 - epsilon) + epsilon 
                         for val in values]
        
        logger.info(f"Normalized {metric_name}: min={min_val:.4f}, max={max_val:.4f}")
        return normalized
    
    def calculate_tts_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall score for TTS systems.
        
        Args:
            metrics: Dictionary containing metric values for a TTS system
                    Keys: DNSMOS, SIM, PSNR, WER, RTF, LogParams
                    
        Returns:
            Overall TTS score (higher is better)
        """
        # Validate required metrics
        required_metrics = set(self.tts_weights.keys())
        available_metrics = set(metrics.keys())
        
        if not required_metrics.issubset(available_metrics):
            missing = required_metrics - available_metrics
            raise ValueError(f"Missing required TTS metrics: {missing}")
        
        # Calculate weighted sum
        score = 0.0
        for metric_name, weight in self.tts_weights.items():
            value = metrics[metric_name]
            score += weight * value
            
        return score
    
    def calculate_audio_reconstruction_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall score for audio reconstruction models.
        
        Args:
            metrics: Dictionary containing metric values for an audio reconstruction model
                    Keys: DNSMOS, PESQ, PSNR, SSIM, RTF, LogParams
                    
        Returns:
            Overall audio reconstruction score (higher is better)
        """
        # Validate required metrics
        required_metrics = set(self.audio_recon_weights.keys())
        available_metrics = set(metrics.keys())
        
        if not required_metrics.issubset(available_metrics):
            missing = required_metrics - available_metrics
            raise ValueError(f"Missing required audio reconstruction metrics: {missing}")
        
        # Calculate weighted sum
        score = 0.0
        for metric_name, weight in self.audio_recon_weights.items():
            value = metrics[metric_name]
            score += weight * value
            
        return score
    
    def evaluate_tts_systems(self, tts_data: List[Dict]) -> pd.DataFrame:
        """
        Evaluate multiple TTS systems and return ranked results.
        
        Args:
            tts_data: List of dictionaries, each containing:
                     - 'name': System name
                     - 'DNSMOS': DNSMOS score
                     - 'SIM': Speaker similarity score
                     - 'PSNR': Peak signal-to-noise ratio
                     - 'WER': Word error rate (will be inverted)
                     - 'RTF': Real-time factor (will be inverted)
                     - 'LogParams': Log of parameter count (will be inverted)
                     
        Returns:
            DataFrame with ranked TTS systems and their scores
        """
        if not tts_data:
            return pd.DataFrame()
        
        # Extract all metric values for normalization
        all_dnsmos = [item['DNSMOS'] for item in tts_data]
        all_sim = [item['SIM'] for item in tts_data]
        all_psnr = [item['PSNR'] for item in tts_data]
        all_wer = [item['WER'] for item in tts_data]
        all_rtf = [item['RTF'] for item in tts_data]
        all_logparams = [item['LogParams'] for item in tts_data]
        
        # Normalize metrics (invert WER, RTF, LogParams since lower is better)
        norm_dnsmos = self.normalize_metric(all_dnsmos, 'DNSMOS')
        norm_sim = self.normalize_metric(all_sim, 'SIM')
        norm_psnr = self.normalize_metric(all_psnr, 'PSNR')
        norm_wer = self.normalize_metric(all_wer, 'WER')  # Will be inverted
        norm_rtf = self.normalize_metric(all_rtf, 'RTF')  # Will be inverted
        norm_logparams = self.normalize_metric(all_logparams, 'LogParams')  # Will be inverted
        
        # Invert metrics where lower is better
        norm_wer = [1.0 - val for val in norm_wer]
        norm_rtf = [1.0 - val for val in norm_rtf]
        norm_logparams = [1.0 - val for val in norm_logparams]
        
        # Calculate overall scores
        results = []
        for i, item in enumerate(tts_data):
            normalized_metrics = {
                'DNSMOS': norm_dnsmos[i],
                'SIM': norm_sim[i],
                'PSNR': norm_psnr[i],
                'WER': norm_wer[i],
                'RTF': norm_rtf[i],
                'LogParams': norm_logparams[i]
            }
            
            overall_score = self.calculate_tts_score(normalized_metrics)
            
            results.append({
                'Name': item['name'],
                'Overall_Score': overall_score,
                'DNSMOS': item['DNSMOS'],
                'SIM': item['SIM'],
                'PSNR': item['PSNR'],
                'WER': item['WER'],
                'RTF': item['RTF'],
                'LogParams': item['LogParams'],
                'DNSMOS_Norm': norm_dnsmos[i],
                'SIM_Norm': norm_sim[i],
                'PSNR_Norm': norm_psnr[i],
                'WER_Norm': norm_wer[i],
                'RTF_Norm': norm_rtf[i],
                'LogParams_Norm': norm_logparams[i]
            })
        
        # Create DataFrame and sort by overall score
        df = pd.DataFrame(results)
        df = df.sort_values('Overall_Score', ascending=False)
        
        return df
    
    def evaluate_audio_reconstruction_systems(self, audio_data: List[Dict]) -> pd.DataFrame:
        """
        Evaluate multiple audio reconstruction systems and return ranked results.
        
        Args:
            audio_data: List of dictionaries, each containing:
                       - 'name': System name
                       - 'DNSMOS': DNSMOS score
                       - 'PESQ': PESQ score
                       - 'PSNR': Peak signal-to-noise ratio
                       - 'SSIM': Structural similarity index
                       - 'RTF': Real-time factor (will be inverted)
                       - 'LogParams': Log of parameter count (will be inverted)
                       
        Returns:
            DataFrame with ranked audio reconstruction systems and their scores
        """
        if not audio_data:
            return pd.DataFrame()
        
        # Extract all metric values for normalization
        all_dnsmos = [item['DNSMOS'] for item in audio_data]
        all_pesq = [item['PESQ'] for item in audio_data]
        all_psnr = [item['PSNR'] for item in audio_data]
        all_ssim = [item['SSIM'] for item in audio_data]
        all_rtf = [item['RTF'] for item in audio_data]
        all_logparams = [item['LogParams'] for item in audio_data]
        
        # Normalize metrics (invert RTF, LogParams since lower is better)
        norm_dnsmos = self.normalize_metric(all_dnsmos, 'DNSMOS')
        norm_pesq = self.normalize_metric(all_pesq, 'PESQ')
        norm_psnr = self.normalize_metric(all_psnr, 'PSNR')
        norm_ssim = self.normalize_metric(all_ssim, 'SSIM')
        norm_rtf = self.normalize_metric(all_rtf, 'RTF')  # Will be inverted
        norm_logparams = self.normalize_metric(all_logparams, 'LogParams')  # Will be inverted
        
        # Invert metrics where lower is better
        norm_rtf = [1.0 - val for val in norm_rtf]
        norm_logparams = [1.0 - val for val in norm_logparams]
        
        # Calculate overall scores
        results = []
        for i, item in enumerate(audio_data):
            normalized_metrics = {
                'DNSMOS': norm_dnsmos[i],
                'PESQ': norm_pesq[i],
                'PSNR': norm_psnr[i],
                'SSIM': norm_ssim[i],
                'RTF': norm_rtf[i],
                'LogParams': norm_logparams[i]
            }
            
            overall_score = self.calculate_audio_reconstruction_score(normalized_metrics)
            
            results.append({
                'Name': item['name'],
                'Overall_Score': overall_score,
                'DNSMOS': item['DNSMOS'],
                'PESQ': item['PESQ'],
                'PSNR': item['PSNR'],
                'SSIM': item['SSIM'],
                'RTF': item['RTF'],
                'LogParams': item['LogParams'],
                'DNSMOS_Norm': norm_dnsmos[i],
                'PESQ_Norm': norm_pesq[i],
                'PSNR_Norm': norm_psnr[i],
                'SSIM_Norm': norm_ssim[i],
                'RTF_Norm': norm_rtf[i],
                'LogParams_Norm': norm_logparams[i]
            })
        
        # Create DataFrame and sort by overall score
        df = pd.DataFrame(results)
        df = df.sort_values('Overall_Score', ascending=False)
        
        return df
    
    def generate_report(self, tts_results: pd.DataFrame, audio_results: pd.DataFrame) -> str:
        """
        Generate a comprehensive report of generator evaluation results.
        
        Args:
            tts_results: DataFrame with TTS system results
            audio_results: DataFrame with audio reconstruction results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("GENERATOR OVERALL SCORE EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # TTS Systems Report
        if not tts_results.empty:
            report.append("TTS SYSTEMS RANKINGS:")
            report.append("-" * 30)
            for i, (_, row) in enumerate(tts_results.iterrows(), 1):
                report.append(f"{i:2d}. {row['Name']:20s} - Score: {row['Overall_Score']:.4f}")
                report.append(f"     DNSMOS: {row['DNSMOS']:.3f}, SIM: {row['SIM']:.3f}, PSNR: {row['PSNR']:.2f}")
                report.append(f"     WER: {row['WER']:.3f}, RTF: {row['RTF']:.3f}, LogParams: {row['LogParams']:.2f}")
                report.append("")
        
        # Audio Reconstruction Systems Report
        if not audio_results.empty:
            report.append("AUDIO RECONSTRUCTION SYSTEMS RANKINGS:")
            report.append("-" * 40)
            for i, (_, row) in enumerate(audio_results.iterrows(), 1):
                report.append(f"{i:2d}. {row['Name']:20s} - Score: {row['Overall_Score']:.4f}")
                report.append(f"     DNSMOS: {row['DNSMOS']:.3f}, PESQ: {row['PESQ']:.3f}, PSNR: {row['PSNR']:.2f}")
                report.append(f"     SSIM: {row['SSIM']:.3f}, RTF: {row['RTF']:.3f}, LogParams: {row['LogParams']:.2f}")
                report.append("")
        
        # Weight information
        report.append("METRIC WEIGHTS:")
        report.append("-" * 15)
        report.append("TTS Systems:")
        for metric, weight in self.tts_weights.items():
            report.append(f"  {metric}: {weight:.3f}")
        report.append("")
        report.append("Audio Reconstruction:")
        for metric, weight in self.audio_recon_weights.items():
            report.append(f"  {metric}: {weight:.3f}")
        
        return "\n".join(report)


def load_generator_data_from_csv(tts_file: str = None, audio_file: str = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Load generator evaluation data from CSV files.
    
    Args:
        tts_file: Path to TTS systems CSV file
        audio_file: Path to audio reconstruction systems CSV file
        
    Returns:
        Tuple of (tts_data, audio_data) lists
    """
    tts_data = []
    audio_data = []
    
    if tts_file:
        try:
            df = pd.read_csv(tts_file)
            for _, row in df.iterrows():
                tts_data.append({
                    'name': row['Name'],
                    'DNSMOS': row['DNSMOS'],
                    'SIM': row['SIM'],
                    'PSNR': row['PSNR'],
                    'WER': row['WER'],
                    'RTF': row['RTF'],
                    'LogParams': row['LogParams']
                })
            logger.info(f"Loaded {len(tts_data)} TTS systems from {tts_file}")
        except Exception as e:
            logger.error(f"Error loading TTS file {tts_file}: {e}")
    
    if audio_file:
        try:
            df = pd.read_csv(audio_file)
            for _, row in df.iterrows():
                audio_data.append({
                    'name': row['Name'],
                    'DNSMOS': row['DNSMOS'],
                    'PESQ': row['PESQ'],
                    'PSNR': row['PSNR'],
                    'SSIM': row['SSIM'],
                    'RTF': row['RTF'],
                    'LogParams': row['LogParams']
                })
            logger.info(f"Loaded {len(audio_data)} audio reconstruction systems from {audio_file}")
        except Exception as e:
            logger.error(f"Error loading audio file {audio_file}: {e}")
    
    return tts_data, audio_data


def main():
    """Example usage of the GeneratorOverallScore class."""
    
    # Example TTS data 
    example_tts_data = [
        {
            'name': 'Matcha-TTS',
            'DNSMOS': 4.52,
            'SIM': 0.89,
            'PSNR': 28.45,
            'WER': 0.023,
            'RTF': 0.15,
            'LogParams': 5.2
        },
        {
            'name': 'YourTTS',
            'DNSMOS': 4.38,
            'SIM': 0.92,
            'PSNR': 27.89,
            'WER': 0.031,
            'RTF': 0.12,
            'LogParams': 4.47
        },
        {
            'name': 'MaskGCT-TTS',
            'DNSMOS': 4.45,
            'SIM': 0.94,
            'PSNR': 29.12,
            'WER': 0.018,
            'RTF': 0.25,
            'LogParams': 7.35
        }
    ]
    
    # Example audio reconstruction data
    example_audio_data = [
        {
            'name': 'BigVGAN',
            'DNSMOS': 4.68,
            'PESQ': 4.25,
            'PSNR': 32.45,
            'SSIM': 0.95,
            'RTF': 0.08,
            'LogParams': 6.12
        },
        {
            'name': 'Vocos',
            'DNSMOS': 4.52,
            'PESQ': 4.18,
            'PSNR': 31.78,
            'SSIM': 0.93,
            'RTF': 0.10,
            'LogParams': 5.89
        },
        {
            'name': 'HiFi-GAN',
            'DNSMOS': 4.35,
            'PESQ': 3.95,
            'PSNR': 30.12,
            'SSIM': 0.91,
            'RTF': 0.05,
            'LogParams': 4.23
        }
    ]
    
    # Initialize calculator
    calculator = GeneratorOverallScore()
    
    # Evaluate TTS systems
    tts_results = calculator.evaluate_tts_systems(example_tts_data)
    
    # Evaluate audio reconstruction systems
    audio_results = calculator.evaluate_audio_reconstruction_systems(example_audio_data)
    
    # Generate and print report
    report = calculator.generate_report(tts_results, audio_results)
    print(report)
    
    # Save results
    tts_results.to_csv('tts_generator_scores.csv', index=False)
    audio_results.to_csv('audio_reconstruction_scores.csv', index=False)
    print(f"\nResults saved to 'tts_generator_scores.csv' and 'audio_reconstruction_scores.csv'")


if __name__ == "__main__":
    main() 