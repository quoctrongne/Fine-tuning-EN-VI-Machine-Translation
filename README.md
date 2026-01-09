# Vietnamese-English Neural Machine Translation: Encoder-Decoder vs Decoder-Only Architectures

## 📋 Tổng quan

Dự án này thực hiện nghiên cứu so sánh hiệu suất giữa hai kiến trúc mô hình ngôn ngữ trong tác vụ dịch máy Tiếng Việt - Tiếng Anh:
- **Encoder-Decoder**: ViT5 (Vietnamese T5)
- **Decoder-Only**: Qwen2.5-0.5B-Instruct

Nghiên cứu đánh giá khả năng dịch thuật của cả hai kiến trúc trên hai tập dữ liệu: dữ liệu gốc và dữ liệu đã được tăng cường (augmented), sử dụng các metrics BLEU và ROUGE.

---

## 🎯 Mục tiêu nghiên cứu

1. So sánh hiệu suất giữa kiến trúc Encoder-Decoder và Decoder-Only
2. Đánh giá tác động của data augmentation lên chất lượng dịch
3. Phân tích ưu nhược điểm của từng kiến trúc trong tác vụ dịch máy
4. Cung cấp insights về việc lựa chọn kiến trúc phù hợp cho Vietnamese NMT

---

## 🏗️ Kiến trúc mô hình

### 1. ViT5 (Encoder-Decoder)
- **Base Model**: VietAI/vit5-base
- **Kiến trúc**: Transformer Encoder-Decoder
- **Tham số**: ~223M parameters
- **Đặc điểm**: 
  - Thiết kế chuyên biệt cho sequence-to-sequence tasks
  - Encoder xử lý input, Decoder sinh output
  - Pre-trained trên corpus tiếng Việt lớn

### 2. Qwen2.5-0.5B (Decoder-Only)
- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct
- **Kiến trúc**: Transformer Decoder-Only
- **Tham số**: ~500M parameters
- **Kỹ thuật tối ưu**: LoRA (Low-Rank Adaptation)
  - Rank: 8
  - Alpha: 16
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj
- **Đặc điểm**:
  - Mô hình ngôn ngữ tự hồi quy
  - Fine-tuned với instruction format
  - Efficient training với PEFT

---

## 📊 Dataset

### Nguồn dữ liệu
- **Training Set**: 133,317 cặp câu Vi-En
- **Validation Set**: 1,553 cặp câu
- **Test Set**: 1,268 cặp câu

### Data Augmentation
Áp dụng các kỹ thuật tăng cường dữ liệu:
- **Synonym Replacement**: Thay thế từ đồng nghĩa
- **Random Insertion**: Chèn từ ngẫu nhiên
- **Random Swap**: Hoán đổi vị trí từ
- **Random Deletion**: Xóa từ ngẫu nhiên
- **Tỷ lệ augmentation**: 30% dữ liệu training

### Preprocessing
- Chuẩn hóa Unicode (NFC)
- Loại bỏ ký tự đặc biệt và khoảng trắng thừa
- Tokenization với BPE (Byte-Pair Encoding)
- Max sequence length: 128 tokens

---

## 🔧 Cấu hình Training

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

### Qwen Training (với LoRA)
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

## 📈 Kết quả thực nghiệm

### Bảng so sánh tổng quan

| Model | Dataset | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|---------|------|---------|---------|---------|
| **ViT5** | Original | 0.4187 | 0.6235 | 0.4012 | 0.5789 |
| **ViT5** | Augmented | 0.4523 | 0.6512 | 0.4389 | 0.6087 |
| **Qwen2.5** | Original | 0.3892 | 0.5978 | 0.3745 | 0.5523 |
| **Qwen2.5** | Augmented | 0.4156 | 0.6187 | 0.4023 | 0.5801 |

### Insights chính

#### 1. Hiệu suất tổng thể
- **ViT5 vượt trội** trong tất cả các metrics trên cả hai tập dữ liệu
- Gap hiệu suất: 6-10% giữa ViT5 và Qwen
- Kiến trúc Encoder-Decoder phù hợp hơn cho translation tasks

#### 2. Tác động của Data Augmentation
- **ViT5**: Cải thiện đáng kể
  - BLEU: +0.0336 (+8.0%)
  - ROUGE-L: +0.0298 (+5.1%)
- **Qwen**: Cải thiện vừa phải
  - BLEU: +0.0264 (+6.8%)
  - ROUGE-L: +0.0278 (+5.0%)
- Augmentation có hiệu quả rõ rệt cho cả hai kiến trúc

#### 3. Phân tích chi tiết

**ViT5 Strengths:**
- Precision cao trong semantic preservation
- Fluency tốt nhờ attention mechanism hai chiều
- Stable training convergence

**Qwen Strengths:**
- Khả năng generate creative variations
- Context understanding tốt với long-range dependencies
- Efficient inference với proper prompting

**Trade-offs:**
- ViT5: Yêu cầu nhiều tài nguyên training hơn
- Qwen: Cần prompt engineering cẩn thận
- ViT5: Tốt cho literal translation
- Qwen: Tốt cho contextual adaptation

---

## 🔬 Metrics Evaluation

### BLEU Score
- Đo lường độ chính xác n-gram overlap
- Phù hợp với translation quality assessment
- ViT5 Augmented đạt cao nhất: 0.4523

### ROUGE Scores
- **ROUGE-1**: Unigram overlap (lexical similarity)
- **ROUGE-2**: Bigram overlap (phrase accuracy)
- **ROUGE-L**: Longest common subsequence (structure preservation)
- ViT5 Augmented leading trong tất cả ROUGE variants

---

## 💡 Các phát hiện quan trọng

### 1. Architectural Implications
- Encoder-Decoder architecture có lợi thế rõ ràng cho translation
- Bi-directional attention trong encoder giúp hiểu ngữ cảnh tốt hơn
- Decoder-only models cần more sophisticated prompting

### 2. Data Quality Impact
- Augmentation hiệu quả hơn trên ViT5
- Quality over quantity trong translation training
- Balanced augmentation tránh overfitting

### 3. Resource Efficiency
- LoRA giảm 75% trainable parameters cho Qwen
- ViT5 cần full fine-tuning cho best results
- Trade-off giữa performance và efficiency

---

## 📊 Visualization & Analysis

Notebook bao gồm:
- Training loss curves comparison
- BLEU/ROUGE scores bar charts
- Confusion matrices for error analysis
- Sample translations showcase
- Statistical significance tests

---

## 🎓 Kết luận

### Key Takeaways

1. **Kiến trúc matters**: Encoder-Decoder vẫn là lựa chọn tốt nhất cho translation tasks
2. **Data augmentation works**: Cải thiện 5-8% performance across models
3. **Efficient fine-tuning**: LoRA cho phép train decoder-only models với tài nguyên hạn chế
4. **Context is key**: Cả hai models đều hưởng lợi từ richer contextual information

### Hướng phát triển

- [ ] Thử nghiệm với larger models (ViT5-large, Qwen-1.5B+)
- [ ] Ensemble methods combining both architectures
- [ ] Domain-specific fine-tuning (medical, legal, technical)
- [ ] Multi-task learning (translation + summarization)
- [ ] Low-resource language pairs experiments

---

## 📚 References

- **ViT5**: [VietAI/vit5-base](https://huggingface.co/VietAI/vit5-base)
- **Qwen**: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **BLEU**: [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/)
- **ROUGE**: [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/)

---

## 🤝 Contributing

Contributions are welcome! Hãy mở issue hoặc pull request nếu bạn có ý tưởng cải thiện.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Phát triển như một phần của nghiên cứu về Neural Machine Translation cho tiếng Việt.

---

## 🙏 Acknowledgments

- VietAI team cho ViT5 model
- Alibaba Cloud cho Qwen models
- Hugging Face cho transformers library
- Kaggle/Colab cho GPU resources

---

**Note**: Kết quả có thể vary tùy thuộc vào hardware, random seeds, và hyperparameters. Các số liệu trong README này là từ runs được documented trong notebook.
