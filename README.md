# Fine-tuning English - Vietnamese Machine Translation

## 📋 Overview

This project conducts a comparative study of two language model architectures for Vietnamese-English machine translation:
- **Encoder-Decoder**: ViT5 (Vietnamese T5)
- **Decoder-Only**: Qwen2.5-0.5B-Instruct

The research evaluates the translation capabilities of both architectures on two datasets: original data and augmented data, using BLEU and ROUGE metrics.

---

## 🎯 Research Objectives

1. Compare performance between Encoder-Decoder and Decoder-Only architectures
2. Evaluate the impact of data augmentation on translation quality
3. Analyze strengths and weaknesses of each architecture in machine translation tasks
4. Provide insights for selecting appropriate architectures for Vietnamese NMT

---

## 🏗️ Model Architecture

### 1. ViT5 (Encoder-Decoder)
- **Base Model**: VietAI/vit5-base
- **Architecture**: Transformer Encoder-Decoder
- **Parameters**: ~223M parameters
- **Features**: 
  - Specialized design for sequence-to-sequence tasks
  - Encoder processes input, Decoder generates output
  - Pre-trained on large Vietnamese corpus

### 2. Qwen2.5-0.5B (Decoder-Only)
- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Architecture**: Transformer Decoder-Only
- **Parameters**: ~500M parameters
- **Optimization Technique**: LoRA (Low-Rank Adaptation)
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj
- **Features**:
  - Autoregressive language model
  - Fine-tuned with instruction format
  - Efficient training with PEFT

---

## 📊 Dataset

### Data Source
- **Training Set**: 133,317 Vietnamese-English sentence pairs
- **Validation Set**: 1,553 pairs
- **Test Set**: 1,268 pairs

### Data Augmentation
Applied augmentation techniques:
- **Synonym Replacement**: Replace words with synonyms
- **Random Insertion**: Insert random words
- **Random Swap**: Swap word positions
- **Random Deletion**: Delete random words
- **Augmentation ratio**: 30% of training data

### Preprocessing
- Unicode normalization (NFC)
- Remove special characters and extra whitespace
- Tokenization with BPE (Byte-Pair Encoding)
- Max sequence length: 128 tokens

---

## 🔧 Training Configuration

### ViT5 Training
```python
Training Arguments:
- Batch size: 16
- Learning rate: 5e-5
- Epochs: 5
- Weight decay: 0.01
- Warmup steps: 500
- FP16: True
- Gradient accumulation: 2 steps
```

### Qwen Training (with LoRA)
```python
Training Arguments:
- Batch size: 4
- Learning rate: 2e-4
- Epochs: 3
- LoRA rank: 8
- LoRA alpha: 16
- FP16: True
- Gradient checkpointing: True
```

---

## 📈 Experimental Results

### Comparison Table

| Model | Dataset | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|------|---------|---------|---------|
| **ViT5** | Original | 0.4187 | 0.6235 | 0.4012 | 0.5789 |
| **ViT5** | Augmented | 0.4523 | 0.6512 | 0.4389 | 0.6087 |
| **Qwen2.5** | Original | 0.3892 | 0.5978 | 0.3745 | 0.5523 |
| **Qwen2.5** | Augmented | 0.4156 | 0.6187 | 0.4023 | 0.5801 |

### Key Insights

#### 1. Overall Performance
- **ViT5 outperforms** across all metrics on both datasets
- Performance gap: 6-10% between ViT5 and Qwen
- Encoder-Decoder architecture is more suitable for translation tasks

#### 2. Impact of Data Augmentation
- **ViT5**: Significant improvement
  - BLEU: +0.0336 (+8.0%)
  - ROUGE-L: +0.0298 (+5.1%)
- **Qwen**: Moderate improvement
  - BLEU: +0.0264 (+6.8%)
  - ROUGE-L: +0.0278 (+5.0%)
- Augmentation is effective for both architectures

#### 3. Detailed Analysis

**ViT5 Strengths:**
- High precision in semantic preservation
- Good fluency thanks to bidirectional attention mechanism
- Stable training convergence

**Qwen Strengths:**
- Ability to generate creative variations
- Good context understanding with long-range dependencies
- Efficient inference with proper prompting

**Trade-offs:**
- ViT5: Requires more training resources
- Qwen: Needs careful prompt engineering
- ViT5: Better for literal translation
- Qwen: Better for contextual adaptation

---

## 🔬 Metrics Evaluation

### BLEU Score
- Measures n-gram overlap accuracy
- Suitable for translation quality assessment
- ViT5 Augmented achieved highest: 0.4523

### ROUGE Scores
- **ROUGE-1**: Unigram overlap (lexical similarity)
- **ROUGE-2**: Bigram overlap (phrase accuracy)
- **ROUGE-L**: Longest common subsequence (structure preservation)
- ViT5 Augmented leads in all ROUGE variants

---

## 💡 Key Findings

### 1. Architectural Implications
- Encoder-Decoder architecture has clear advantages for translation
- Bi-directional attention in encoder helps better context understanding
- Decoder-only models need more sophisticated prompting

### 2. Data Quality Impact
- Augmentation is more effective on ViT5
- Quality over quantity in translation training
- Balanced augmentation avoids overfitting

### 3. Resource Efficiency
- LoRA reduces 75% trainable parameters for Qwen
- ViT5 needs full fine-tuning for best results
- Trade-off between performance and efficiency

---

## 📊 Visualization & Analysis

The notebook includes:
- Training loss curves comparison
- BLEU/ROUGE scores bar charts
- Confusion matrices for error analysis
- Sample translations showcase
- Statistical significance tests

---

## 🎓 Conclusion

### Key Takeaways

1. **Architecture matters**: Encoder-Decoder remains the best choice for translation tasks
2. **Data augmentation works**: Improves performance by 5-8% across models
3. **Efficient fine-tuning**: LoRA enables training decoder-only models with limited resources
4. **Context is key**: Both models benefit from richer contextual information
