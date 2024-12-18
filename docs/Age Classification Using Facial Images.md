## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Project Structure](#project-structure)
3. [Age Classes](#age-classes)
4. [Code Implementation](#code-implementation)
5. [Model Comparison](#model-comparison)
6. [Results and Analysis](#results-and-analysis)

## Dataset Overview

The project uses the UTKFace dataset, which is a large-scale face dataset with long age span (range from 0 to 116 years old). The dataset consists of over 20,000 face images with annotations of age, gender, and ethnicity.

### Dataset Characteristics:
- **Source**: [UTKFace Dataset on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)
- **Image Format**: Aligned and cropped faces
- **Resolution**: Original images cropped and resized to 64x64 pixels
- **Color Space**: Converted to grayscale for processing
- **File Naming**: [age]_[gender]_[race]_[date&time].jpg

## Age Classes

For this project, we focused on early childhood age classification (ages 1-4):

- Class 1: 1 year old
- Class 2: 2 years old
- Class 3: 3 years old
- Class 4: 4 years old

### Age Distribution After Augmentation
![[Pasted image 20241218031716.png]]
*Distribution of age classes after data augmentation*

## Code Implementation

### Data Preprocessing

1. **Image Processing**:
   ```python
   # Convert to grayscale and resize
   image = Image.open(image_file).convert('L').resize((64, 64))
   image = np.array(image) / 255.0
   ```

2. **Data Augmentation**:
   - Horizontal flipping
   - Multiple rotation angles (-15°, -10°, -5°, 5°, 10°, 15°)
   - Brightness variations (70%, 85%, 115%, 130%)

3. **Feature Engineering**:
   - StandardScaler for normalization
   - PCA dimensionality reduction (98% variance retention)

### PCA Analysis
![[Pasted image 20241218031741.png]]
*Cumulative explained variance ratio by PCA components*

## Model Comparison

### Model Architectures

1. **K-Nearest Neighbors (KNN)**:
   - Hyperparameters optimized:
     - n_neighbors: [3, 4, 5, 6]
     - weights: ['uniform', 'distance']
     - metric: ['manhattan', 'euclidean']

2. **Logistic Regression**:
   - Hyperparameters optimized:
     - C: [0.1, 1.0, 10.0]
     - solver: ['lbfgs', 'newton-cg']
     - max_iter: 1000

### Performance Metrics

| Metric            | KNN      | Logistic Regression |
| ----------------- | -------- | ------------------- |
| Accuracy          | 0.526932 | 0.545667            |
| Macro-Average AUC | 0.604950 | 0.733457            |
| Log Loss          | 8.764438 | 1.324405            |

### Confusion Matrices

#### KNN Model
![[Pasted image 20241218031804.png]]
*Confusion matrix for KNN classifier*

#### Logistic Regression Model
![[Pasted image 20241218031822.png]]
*Confusion matrix for Logistic Regression classifier*

## Results and Analysis

### Key Findings

1. **Model Performance**:
   - Logistic showed slightly better performance in accuracy and AUC
   - Logistic Regression demonstrated more stable predictions across classes

2. **Feature Importance**:
   - PCA reduced dimensions while maintaining 98% of variance
   - First 150 components captured most significant features

3. **Class Balance**:
   - Data augmentation helped balance class distribution
   - Improved model performance for underrepresented ages

### Comparison Visualization
![[Pasted image 20241218032047.png]]
*Performance comparison between KNN and Logistic Regression*

### Improvements Made

1. **Data Enhancement**:
   - Comprehensive augmentation strategy
   - Balanced class distribution
   - Robust preprocessing pipeline

2. **Model Optimization**:
   - GridSearchCV for hyperparameter tuning
   - Cross-validation for reliable evaluation
   - Multiple evaluation metrics

3. **Results Visualization**:
   - Detailed confusion matrices
   - ROC curves and AUC scores
   - Performance comparison plots

## Future Improvements

1. **Model Enhancements**:
   - Implement deep learning approaches (CNN)
   - Explore ensemble methods
   - Test more advanced augmentation techniques

2. **Feature Engineering**:
   - Investigate facial landmarks
   - Add more domain-specific features
   - Test different dimensionality reduction techniques

3. **Evaluation**:
   - Add cross-age analysis
   - Implement uncertainty quantification
   - Test on external datasets

---