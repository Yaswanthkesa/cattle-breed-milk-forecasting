# 🐄 Cattle Breed Recognition and Milk Yield Forecasting Using Deep Learning

This project presents a **multimodal deep learning system** that integrates **image-based cattle breed classification** with **time-series milk-trait prediction**. The pipeline combines advanced **CNN feature extractors** and **LSTM temporal models** to simultaneously perform:

- **37-breed classification**
- **Next-day prediction of 6 key milk traits**

This hybrid approach enhances the accuracy and reliability of dairy analytics, supporting **precision livestock farming**, **breed conservation**, and **smart dairy management**.

---

## 📘 About the Dataset

The dataset used in this research is a **multimodal dataset** combining:

### 🖼️ 1. **Image Dataset (Phenotypic Features)**  
A curated collection of **2,960 cattle and buffalo images** across **37 Indian breeds**, sourced under real-world farm conditions.

- **Images per breed:** 80  
- **Total images:** 2,960  
- **Image types:** RGB  
- **Variability:** natural lighting, backgrounds, occlusions  
- **Resolution standardized:** `224 × 224`

Each image is mapped to:

- Breed label  
- Absolute file path  
- Unique image ID  

![Sample Image](assets/sample_cow.png)

---

### 🥛 2. **Milk Trait Dataset (Physiological Features)**  

Each breed is associated with **30-day historical milk-composition records**:

| Feature | Description |
|--------|-------------|
| Milk Yield | Daily milk production |
| Fat (%) | Fat concentration |
| Protein (%) | Protein content |
| Lactose (%) | Lactose level |
| SNF | Solids Not Fat |
| Total Solids | Overall solids |

**Daily measurements** are cleaned and normalized using:

\[
x_{\text{scaled}}=\frac{x-x_{\min}}{x_{\max}-x_{\min}}
\]

A final multimodal table of **88,800 samples** is constructed:


---

## 📦 Dataset Details

- **Total Size:** ~3 GB  
- **Number of Breeds:** 37  
- **Total Images:** 2,960  
- **Milk-traits:** 6 features over 30 days  
- **Applications:**  
  ✔ Cattle Breed Identification  
  ✔ Milk Yield Forecasting  
  ✔ Livestock Monitoring  
  ✔ Precision Dairy Farming  

---

## 🧹 Data Preprocessing

### 🖼 Image Preprocessing
- Resized to CNN-specific input shape (`224×224×3`)
- Normalized using ImageNet statistics
- No segmentation (full animal preserved)
- Variations in coat patterns, horn structure, and muzzle features retained

| Step | Description |
|------|-------------|
| Resize | Standardized to 224×224 |
| Normalize | `/255` scaling + ImageNet mean-std |
| Mapping | Linked by Breed & Image ID |

---

### 🥛 Milk Trait Preprocessing
- Invalid entries removed  
- Missing values handled  
- Chronological sorting  
- Sliding-window sequence creation:  

\[
(29 \text{ days input}) \rightarrow (\text{30th day prediction})
\]

---

## 🔧 Methodology

The proposed pipeline combines:

### 🎯 **1. CNN Feature Extraction**
Pretrained CNNs used:

| Model | Purpose |
|-------|---------|
| MobileNetV2 | Best overall accuracy |
| Xception | Strong deep feature extraction |
| DenseNet121 | High gradient flow |
| EfficientNet-B4 | Balanced depth & width |
| VGG16/19 | Baseline comparison |
| ResNet50 | Residual learning |

Each CNN outputs an **image embedding vector** (1,000–2,000 dims).

---

### 🎯 **2. LSTM for Temporal Milk Traits**
Input shape:
(29 days × 6 traits)

The LSTM learns:

- Cyclic patterns  
- Daily fluctuations  
- Trait dependencies  
- Seasonal variations  

Outputs a **128-dimensional temporal embedding**.

---

### 🎯 **3. Multimodal Fusion Layer**

Final representation is:

\[
f_{\text{fusion}} = [f_{\text{img}} \Vert f_{\text{lstm}}]
\]

This fused vector is passed through:

- Dense units  
- BatchNorm  
- Dropout  

---

### 🎯 **4. Multi-Task Output**

#### 🐄 Breed Classification (Softmax)
- 37-class prediction  
- CrossEntropy loss  

#### 🥛 Milk-Trait Regression (Linear)
- Predicts all 6 traits simultaneously  
- MSE loss  

---

![Methodology Pipeline](assets/methodology_pipeline.png)

---

## 🏗️ Model Architecture Summary

| Component | Description |
|-----------|-------------|
| CNN Backbone | Pretrained feature extractor |
| Global Pooling | Converts feature maps → vector |
| LSTM Layer | Learns 29-day temporal patterns |
| Fusion Layer | Concatenates CNN + LSTM embeddings |
| Fully Connected | Shared hidden layers |
| Output Heads | Softmax + Regression |

---

## 🛠️ Model Training

- **Optimizer:** Adam  
- **Batch Size:** 16–32  
- **Epochs:** 30–50  
- **Early Stopping:** Patience = 5  
- **Loss:**  
  - Classification → Sparse Categorical Crossentropy  
  - Regression → Mean Squared Error  

Training command:

```python
model.fit([X_img, X_seq], [y_breed, y_milk])
