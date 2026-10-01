# Spoken English Grammar Scoring Engine

## Overview

This project develops an end-to-end machine-learning system for predicting a **spoken-English grammar score on a 0–5 scale from WAV audio recordings**.

The system treats spoken grammar assessment as a **multimodal regression problem**. It combines:

- Acoustic and prosodic features
- Automatic Speech Recognition (ASR)
- Transcript-based linguistic features
- Grammar and fluency signals
- Pretrained speech representations
- Pretrained text representations
- Multiple regression models
- Out-of-fold model comparison and ensemble selection

The central idea is to model both **how the response is spoken** and **what is being said**, rather than relying on a single representation.

---

## Project Pipeline

```text
WAV Audio
   │
   ├── Audio preprocessing
   │
   ├── Acoustic / prosodic features
   │
   ├── Whisper ASR
   │       │
   │       └── Transcript
   │              ├── Linguistic features
   │              ├── Grammar features
   │              ├── Fluency features
   │              └── Text embeddings
   │
   ├── WavLM / wav2vec2 speech representations
   │
   └── Feature combination
             │
             ▼
      Cross-validated models
             │
             ▼
       Out-of-fold predictions
             │
             ▼
       Ensemble / stacking
             │
             ▼
       Final test prediction
             │
             ▼
        submission.csv
```

---

# 1. Problem Definition

The task is to predict a continuous grammar score for spoken-English responses.

### Input

WAV audio recordings containing spoken English.

### Target

A numerical grammar score in the range:

```text
0 to 5
```

Because the target is numerical and ordered, the task is formulated as **regression**.

The development process uses:

- **RMSE** for numerical prediction error
- **Pearson correlation** for agreement between predicted and actual score variation

---

# 2. Why a Multimodal Approach?

A grammar score cannot be fully represented by raw acoustic statistics alone.

Two speakers can have similar:

- pitch,
- loudness,
- speaking speed,
- spectral characteristics,

while producing very different grammatical structures.

At the same time, a transcript loses information about:

- pauses,
- hesitation,
- pitch,
- energy,
- speaking rhythm,
- timing.

Therefore the project combines several information sources.

### Audio features

Capture:

- energy
- pitch
- spectral characteristics
- pauses
- speech timing
- prosody

### Text features

Capture:

- vocabulary
- sentence structure
- repetition
- fillers
- function words
- sentence length

### Grammar and fluency models

Provide more direct linguistic signals.

### Pretrained representations

Capture higher-level patterns that are difficult to reproduce with handcrafted features.

---

# 3. Dataset Discovery

The notebook works with the competition files available in the Colab runtime.

It discovers:

```text
train.csv
test.csv
sample_submission.csv
*.wav
```

Audio files are searched recursively so nested archive structures can be handled.

The notebook also checks:

- missing audio files
- duplicate filenames
- filename collisions
- audio availability for train/test records
- possible duplicate recordings

The actual `test.csv` is treated as the authoritative set of records for prediction.

---

# 4. Audio Preprocessing

Audio is standardized before feature extraction.

The pipeline uses:

- mono audio
- 16 kHz sample rate
- floating-point waveforms

Preprocessing includes controlled normalization and removal of DC offset.

Very short recordings are padded when required by downstream signal-processing operations.

Aggressive silence removal is avoided because pauses and hesitation may contain useful information for spoken-language assessment.

---

# 5. Acoustic and Prosodic Features

The project extracts compact statistical descriptions of the waveform.

## Energy

Examples include:

- RMS energy
- energy variability
- dynamic range

These describe how speech energy changes throughout a response.

## Spectral features

The pipeline includes:

- zero-crossing rate
- spectral centroid
- spectral bandwidth
- spectral roll-off
- spectral flatness
- spectral contrast

These describe the overall spectral characteristics of the speech.

## MFCC

Mel-frequency cepstral coefficients are extracted together with temporal variation features.

The pipeline uses:

- MFCC statistics
- MFCC deltas

## Pitch

Pitch/F0 features capture intonation and speaking dynamics.

Examples include:

- mean F0
- F0 standard deviation
- F0 percentiles
- F0 range
- F0 variability

## Pause and timing

The pipeline measures:

- speech ratio
- speech segment count
- segments per minute
- leading silence
- trailing silence
- gap duration statistics
- long-pause counts
- speech-segment duration
- onset rate

These features preserve information that disappears after converting speech into text.

---

# 6. Automatic Speech Recognition

The project uses **Whisper/faster-whisper** to convert spoken responses into text.

The ASR stage can retain:

- transcript text
- word timestamps
- word confidence/probability
- segment information
- ASR diagnostics

ASR results are cached to avoid repeating expensive transcription.

Transcript coverage is checked before transcript-dependent modelling is used. This prevents an empty or incomplete transcript cache from producing invalid text features.

---

# 7. Transcript Features

The transcript is converted into numerical linguistic features.

### Lexical features

- word count
- unique-word count
- type-token ratio
- average word length
- long-word ratios

### Sentence features

- sentence count
- words per sentence
- sentence-level statistics

### Repetition

- repeated-token rate

### Fillers

Common spoken fillers and hesitation terms are identified.

### Function words

The pipeline calculates ratios involving:

- function words
- auxiliary verbs
- pronouns

### Timing

ASR word timestamps are used to derive additional speaking-rate and timing information.

---

# 8. Grammar Features

The project includes explicit grammar-oriented features rather than relying only on generic embeddings.

## CoLA grammatical acceptability

A pretrained grammatical acceptability model associated with the **Corpus of Linguistic Acceptability (CoLA)** is used as a feature extractor.

Sentence-level outputs are summarized using statistics such as:

- mean
- weighted mean
- minimum
- percentiles
- fraction below selected thresholds

This provides an explicit grammaticality-related signal.

---

# 9. Language-Model Fluency

A pretrained language model is used to derive sentence-level fluency information.

The pipeline calculates normalized language-model likelihood / negative-log-likelihood style features.

These are summarized at response level using measures such as:

- mean NLL
- NLL variability
- upper percentiles
- maximum NLL
- perplexity-related statistics

These features are treated as complementary evidence rather than as a direct grammar label.

---

# 10. LanguageTool Grammar Analysis

LanguageTool is used as an additional rule-based grammar signal when available.

The pipeline derives counts and normalized rates for grammar-related matches.

Examples include:

- agreement
- verb/tense
- article/determiner
- plural
- preposition/order
- confusion/homophone patterns

Style-only and formatting-related warnings are separated where possible to reduce their influence on grammar scoring.

---

# 11. Pretrained Speech Representations

The project uses pretrained self-supervised speech models such as:

```text
microsoft/wavlm-base-plus
facebook/wav2vec2-base-960h
```

Hidden representations are pooled over time.

The representation can retain multiple statistics such as:

- temporal mean
- temporal standard deviation

Different transformer layers can also be evaluated because different layers may encode different levels of acoustic, phonetic and linguistic information.

These representations provide information that handcrafted acoustic features may not capture.

---

# 12. Pretrained Text Embeddings

The transcript is converted into a dense sentence representation using a pretrained sentence-transformer model.

This captures higher-level linguistic structure that simple counts cannot represent.

For example, two responses can have similar:

- word count
- vocabulary size
- filler rate

while having different sentence organization.

Dense text embeddings provide a way to represent these differences.

---

# 13. Feature Groups

The overall feature space consists of complementary groups.

| Feature group | Main information |
|---|---|
| Acoustic | Energy and spectral characteristics |
| Prosodic | Pitch, timing and speaking dynamics |
| Pause | Hesitation and silence structure |
| ASR | Recognition confidence and timing |
| Linguistic | Vocabulary and sentence structure |
| Grammar | Grammatical acceptability and rule signals |
| Fluency | Language-model likelihood |
| Speech embeddings | Learned speech representation |
| Text embeddings | Learned language representation |

Keeping the groups separate makes ablation and model comparison possible.

---

# 14. Validation Strategy

The labelled dataset is relatively small, so a single train/validation split can produce an unstable estimate.

Cross-validation is therefore used.

The validation strategy aims to:

- preserve approximate target distribution
- keep duplicate audio together
- reduce leakage
- generate reliable out-of-fold predictions

Audio hashes can be used to identify identical recordings and keep them in the same validation group.

---

# 15. Leakage Prevention

The modelling pipeline follows several leakage-prevention principles:

1. Test labels are never used.
2. Validation records are not used to fit preprocessing transformations.
3. Feature selection is performed inside training folds.
4. Duplicate recordings are grouped where possible.
5. Ensemble weights are derived from out-of-fold predictions.
6. No manual labels are assigned to validation/test records.

This makes the local validation result more meaningful.

---

# 16. Regression Models

Several model families are evaluated.

## Ridge Regression

A stable linear model for high-dimensional correlated features.

Useful for:

- TF-IDF features
- embeddings
- engineered features

Regularization helps control coefficient instability.

## Support Vector Regression

Included because nonlinear kernels can learn smooth relationships that linear regression cannot capture.

## Extra Trees

A nonlinear tree ensemble capable of modelling interactions among acoustic and linguistic features.

## Random Forest

Provides another tree-based modelling perspective and adds model diversity.

## Gradient Boosting

Gradient-boosted tree models are used to capture nonlinear relationships and feature interactions.

## CatBoost

CatBoost provides another regularized nonlinear learner for multimodal tabular features when available.

---

# 17. Feature Ablation

Feature ablation is used to understand which feature families contribute predictive information.

The project can compare combinations such as:

```text
Acoustic
Text / linguistic
Grammar
Speech representations
Text representations
Combined features
```

Ablation is performed using the same validation methodology so that feature groups can be compared consistently.

---

# 18. Model Selection

Candidate models produce out-of-fold predictions.

These predictions are used to compare:

- individual model RMSE
- Pearson correlation
- prediction correlation
- feature-group contribution
- ensemble performance

A model does not need to be the single strongest model to be useful. A model with different errors can improve an ensemble.

---

# 19. Out-of-Fold Ensemble

The final modelling strategy evaluates several ensemble approaches.

Possible strategies include:

- best individual model
- mean of strong models
- top-k model average
- positive linear stacking
- restricted stacking

The stacking model is trained on out-of-fold predictions rather than on predictions produced from the same records used to fit the base models.

This reduces the risk of overfitting ensemble weights.

---

# 20. Evaluation Metrics

## RMSE

```text
RMSE = sqrt(mean((y_true - y_pred)^2))
```

Lower RMSE indicates smaller prediction errors.

## Pearson correlation

Pearson correlation measures how strongly predicted scores track actual score variation.

The metrics provide complementary information:

- RMSE measures numerical error.
- Pearson measures score-variation agreement.

---

# 21. Error Analysis

Out-of-fold predictions are analyzed to identify systematic weaknesses.

The notebook can examine:

- actual vs predicted scores
- residual distribution
- error by score range
- recording duration vs error
- transcript length vs error
- ASR confidence vs error
- prediction-vs-target slope

The largest validation errors can also be inspected alongside transcript snippets.

This helps identify issues such as:

- regression toward the mean
- short-response instability
- ASR failures
- errors on extreme scores
- recording-quality effects

No manual test or validation labels are created during this analysis.

---

# 22. Final Training and Prediction

After model selection:

1. Selected models are refit using all labelled training records.
2. The complete training set is used.
3. Test records are passed through the same feature pipeline.
4. Selected ensemble weights are applied.
5. Predictions are clipped to the valid 0–5 range.

---

# 23. Submission

The final submission is constructed directly from `test.csv`.

Integrity checks verify:

- prediction count
- filename order
- filename uniqueness
- finite predictions
- valid score range

The final file is:

```text
/content/submission.csv
```

---

# 24. Why This Approach Is Stronger Than an Acoustic-Only Baseline

An acoustic-only model sees properties of the waveform, but grammar is fundamentally linguistic.

The final system adds:

```text
Audio
  ↓
Prosody + timing + pitch
  +
ASR transcript
  ↓
Vocabulary + sentence structure
  +
Grammar models
  ↓
Acceptability + grammar rules
  +
Language model
  ↓
Fluency
  +
Pretrained speech/text encoders
  ↓
Higher-level representations
```

The ensemble can therefore use complementary information rather than depending on a single proxy for grammar quality.

---

# 25. Why Transfer Learning Is Used

The labelled competition dataset is small compared with the datasets normally required to train modern speech and language models from scratch.

Training a large neural network only on the competition labels would increase overfitting risk.

Instead, pretrained models are used as representation/feature extractors, while the competition data is used to learn the final task-specific regression mapping.

Conceptually:

```text
Large pretrained model
        ↓
General speech/language representation
        ↓
Competition-specific regression
        ↓
Grammar score
```

This allows the project to use pretrained knowledge without introducing an external labelled competition dataset.

---

# 26. Competition Compliance

The project is designed to respect the competition restrictions.

The methodology does not introduce an external labelled dataset.

Pretrained external models are used as representation or feature extractors.

Validation and test records are not manually labelled.

Test targets are not used for model fitting or feature selection.

Model selection is based on training data and out-of-fold validation predictions.

The final submission is generated programmatically.

---

# 27. Reproducibility

The notebook uses explicit configuration and random seeds.

Expensive intermediate computations such as ASR and pretrained embeddings are cached where possible.

Important reproducibility components include:

- fixed random seed
- explicit cross-validation configuration
- explicit model hyperparameters
- cached ASR results
- cached feature representations
- submission integrity checks

---

# 28. Requirements

The project is designed for Google Colab.

A GPU runtime is recommended because pretrained speech and language models are computationally expensive.

Typical dependencies include:

```text
Python 3
PyTorch
librosa
NumPy
pandas
scikit-learn
faster-whisper
transformers
sentence-transformers
LightGBM
CatBoost
LanguageTool
```

The notebook contains its environment setup.

---

# 29. Running the Notebook

### 1. Open the notebook in Google Colab

### 2. Enable GPU

```text
Runtime → Change runtime type → GPU
```

### 3. Make the competition data available in `/content`

The notebook searches for the competition ZIP or an already extracted dataset.

### 4. Run the notebook from the beginning

The complete workflow is:

```text
Data discovery
      ↓
Audio validation
      ↓
Audio preprocessing
      ↓
Acoustic/prosodic features
      ↓
Whisper ASR
      ↓
Linguistic + grammar + fluency features
      ↓
Pretrained speech/text representations
      ↓
Feature assembly
      ↓
Cross-validation
      ↓
Model comparison
      ↓
OOF ensemble selection
      ↓
Final training
      ↓
Test prediction
      ↓
Submission validation
      ↓
submission.csv
```

---

# 30. Output

The primary competition output is:

```text
/content/submission.csv
```

The file can be downloaded from the Colab file browser and submitted to the competition.

---

# 31. Key Design Decisions

| Decision | Reason |
|---|---|
| Regression | Target is a numerical 0–5 score |
| Whisper ASR | Converts speech into linguistic information |
| Acoustic features | Preserve information lost by transcription |
| Prosodic features | Capture pitch, timing and speaking dynamics |
| Grammar model | Adds grammatical acceptability information |
| Language model | Adds a complementary fluency signal |
| LanguageTool | Adds rule-based grammar information |
| WavLM / wav2vec2 | Provides learned speech representations |
| Text embeddings | Capture higher-level linguistic structure |
| Cross-validation | Reduces dependence on one split |
| Grouped validation | Helps prevent duplicate-audio leakage |
| OOF stacking | Supports leakage-aware ensemble selection |
| Prediction clipping | Enforces the valid 0–5 score range |
| test.csv ordering | Keeps submission rows aligned with test records |

---

# 32. Limitations

### ASR errors

Incorrect transcription can affect downstream linguistic and grammar features.

### Small labelled dataset

Validation estimates can have noticeable variance.

### Domain shift

Public leaderboard results may not perfectly represent private evaluation data.

### Grammar vs fluency

Some features measure fluency or speaking style rather than grammar directly. The ensemble is therefore designed to combine several complementary signals.

### Pretrained-model bias

Pretrained models were not trained specifically for this competition's scoring rubric, so their outputs are used as features rather than assumed to be the final score.

---

# 33. End-to-End Summary

The final system can be summarized as:

```text
Spoken English WAV
        ↓
Audio preprocessing
        ↓
┌───────────────────────────────┐
│ Acoustic / Prosodic Features  │
└───────────────────────────────┘
        +
┌───────────────────────────────┐
│ Whisper ASR                   │
│   ↓                           │
│ Linguistic Features           │
│ Grammar Features              │
│ Fluency Features              │
│ Text Embeddings               │
└───────────────────────────────┘
        +
┌───────────────────────────────┐
│ Pretrained Speech Embeddings  │
└───────────────────────────────┘
        ↓
Feature matrices
        ↓
Leakage-aware cross-validation
        ↓
Multiple regression models
        ↓
Out-of-fold predictions
        ↓
Ensemble / stacking
        ↓
Final test prediction
        ↓
0–5 validation
        ↓
submission.csv
```

---

## Conclusion

The Spoken English Grammar Scoring Engine combines signal processing, automatic speech recognition, linguistic analysis, grammar modelling, pretrained speech representations, pretrained language representations and ensemble regression.

The core principle is to avoid treating any single feature family as a perfect representation of grammar quality.

Instead, the system combines:

**how the response sounds + what was said + grammatical acceptability + fluency + learned speech/language representations**

and uses cross-validation and out-of-fold predictions to select and combine complementary models.

The result is a complete pipeline from raw spoken audio to competition-ready grammar-score predictions.
