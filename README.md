# SoK: Benchmarking Fake Voice Detection in the Fake Voice Generation Arms Race

This repository contains the implementation and evaluation framework for our comprehensive cross-domain assessment of fake voice detection systems. Our work evaluates 8 state-of-the-art fake voice detectors against 20 different fake voice generators, revealing critical vulnerabilities in current detection systems.

## Overview

This study presents the first large-scale, cross-domain evaluation of fake voice detectors, benchmarking 8 state-of-the-art models against datasets synthesized by 20 different fake voices. Our evaluation reveals substantial security vulnerabilities in current fake voice detection systems and provides actionable recommendations for building more resilient technologies.

## Repository Structure

```
benchmark_voice/
├── Baseline-AASIST/          # AASIST baseline implementation
├── Baseline-RawNet2/         # RawNet2 baseline implementation  
├── evaluation-package/       # Core evaluation framework
│   ├── calculate_metrics.py  # Metric calculation for Track 1
│   ├── calculate_metrics_full.py # Extended metric calculations
│   ├── evaluation.py         # Main evaluation script
│   ├── evaluation_full.py    # Full evaluation pipeline
│   ├── detector_overall_score.py # Proposed detector overall score (with samples)
│   ├── generator_overall_score.py # Proposed generator overall score (with samples)
│   └── util.py              # Utility functions
├── Tool-score-fusion/        # Score fusion utilities
└── README.md                # This file
```

## Key Contributions

1. **Comprehensive Overview**: Taxonomy of fake voice generation and detection systems through a security lens
2. **Refined Taxonomy**: Enhanced detector classification with generalization-optimized models
3. **Empirical Security Assessment**: Large-scale cross-domain evaluation with explainable analysis
4. **Recommended Practices**: Concrete guidelines for secure fake voice detection systems

## Quick Start

### Prerequisites

```bash
# Python 3.8+ required
conda create --name fake_voice_benchmark python=3.8
conda activate fake_voice_benchmark

# Install core dependencies
pip install scipy numpy pandas torch torchaudio
```

### Running Cross-Domain Evaluation

1. **Prepare your score files** in the required format (see [Score File Format](#score-file-format) below)

2. **Run Track 1 evaluation** (Countermeasure evaluation):
```bash
cd evaluation-package
python evaluation.py --m t1 --cm your_cm_scores.txt --cm_keys your_cm_keys.txt
```

3. **Run Track 1 evaluation** (Countermeasure evaluation):
```bash
python evaluation.py --m t1 --cm your_cm_scores.txt --cm_keys your_cm_keys.txt
```

## Score File Format

### Track 1 (Countermeasure) Format

**CM Score File** (`cm_scores.txt`):
```
filename	cm-score
E_000001	0.01
E_000002	0.02
E_000003	0.95
```

**CM Key File** (`cm_keys.txt`):
```
filename	cm-label
E_000001	bonafide
E_000002	spoof
E_000003	bonafide
```

## Evaluation Metrics

### Track 1 Metrics (Countermeasure Evaluation)
- **Primary**: minDCF (minimum Detection Cost Function)
- **Secondary**: CLLR (Cost of Log Likelihood Ratio), EER (Equal Error Rate), actDCF (actual Detection Cost Function)

## Cross-Domain Evaluation Framework

Our evaluation framework supports comprehensive cross-domain assessment:

### Fake Voice Generators Evaluated
- **Traditional Pipeline Models**: Tacotron2, FastSpeech2, DelightfulTTS
- **Neural Codec Language Models**: VALL-E, NaturalSpeech2/3, XTTS, MaskGCT-TTS
- **End-to-End Models**: VITS, YourTTS, Mega-TTS
- **Vocoders**: HiFi-GAN, BigVGAN, Vocos, DiffWave, RFWave

### Fake Voice Detectors Evaluated
- **End-to-End Models**: RawNet2, RawPCDart, RawBMamba, AASIST
- **Wav2Vec-2.0 Based**: XLSR-SLS, XLSR-Conformer, XLSR-Conformer+TCM, XLSR-AASIST

### Cross-Domain Scenarios
1. **In-domain evaluation**: Standard ASVspoof datasets (ASVspoof-21LA)
2. **Cross-compression evaluation**: ASVspoof-21DF (compressed audio)
3. **Cross-degradation evaluation**: FoR dataset (waveform degradations)
4. **Cross-lingual evaluation**: CFAD dataset (Chinese)
5. **Cross-generator evaluation**: Individual generator performance analysis

## Reproducing Our Experiments

### 1. Data Preparation
```bash
# Download LibriTTS 
 https://www.openslr.org/60/

# Generate fake voices using our evaluated generators
# (See individual generator repositories for specific instructions)
```

### 2. Detector Evaluation
```bash
# Example: Evaluate XLSR-SLS detector against all generators
for generator in generator_list; do
    python evaluation.py --m t1 \
        --cm scores/${detector}_${generator}_scores.txt \
        --cm_keys keys/${generator}_keys.txt
done
```

### 3. Cross-Domain Analysis
```bash
# Run full cross-domain evaluation using the evaluation package
python evaluation_full.py --config cross_domain_config.yaml
```

## 📊 Results and Analysis

Our evaluation reveals several key findings:

1. **Generator Sophistication Gap**: Advanced neural codec models (MaskGCT-TTS, FireRedTTS-1S) achieve EERs >60% against many detectors
2. **Detector Specialization**: No single detector achieves universal robustness across all generator types
3. **Calibration Issues**: Many detectors show poor score calibration under distribution shifts
4. **Cross-lingual Vulnerability**: Performance degrades significantly on non-English datasets

## 🔧 Advanced Usage

### Custom Metric Calculation
```python
from evaluation-package.calculate_modules import calculate_minDCF_EER_CLLR_actDCF

# Calculate metrics programmatically
minDCF, eer, cllr, actDCF = calculate_minDCF_EER_CLLR_actDCF(
    cm_scores=your_scores,
    cm_keys=your_keys,
    output_file="results.txt"
)
```

### Proposed Overall Score Calculations
Our evaluation package includes implementations of the proposed overall score metrics from the paper:

**Detector Overall Score** (`detector_overall_score.py`):
```python
from evaluation-package.detector_overall_score import DetectorOverallScore

# Initialize calculator with trade-off factor α=0.8
calculator = DetectorOverallScore(alpha=0.8)

# Set generator challenge scores and detector parameters
calculator.set_generator_challenge_scores(challenge_scores)
calculator.set_detector_parameters(detector_params)

# Calculate overall scores combining empirical performance and model complexity
overall_scores = calculator.calculate_overall_scores(eer_metrics, cllr_metrics, mindcf_metrics)
```

**Generator Overall Score** (`generator_overall_score.py`):
```python
from evaluation-package.generator_overall_score import GeneratorOverallScore

# Initialize calculator
calculator = GeneratorOverallScore()

# Evaluate TTS systems using weighted aggregation of normalized metrics
tts_results = calculator.evaluate_tts_systems(tts_data)

# Evaluate audio reconstruction systems
audio_results = calculator.evaluate_audio_reconstruction_systems(audio_data)
```

*Note: Both files contain sample implementations for demonstration purposes. 

### Batch Evaluation
```bash
# Evaluate multiple detectors against multiple generators
python batch_evaluation.py --detectors detector_list.txt --generators generator_list.txt
```


