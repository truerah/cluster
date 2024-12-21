# Clustering Analysis

This project demonstrates a comprehensive clustering analysis using the Wine Quality dataset from the UCI Machine Learning Repository. The primary objective is to evaluate the impact of various clustering algorithms and preprocessing techniques on clustering performance. Through this, the project provides insights into how these methods influence unsupervised learning outcomes.

## Dataset
The Wine Quality dataset comprises a range of physicochemical properties of wine samples, such as alcohol content, pH levels, and acidity. These features are used for clustering without considering the quality score column.




## Methodology
The project applies multiple clustering algorithms (K-Means, Hierarchical Clustering, and Mean Shift) combined with different preprocessing methods to analyze their effectiveness. Evaluation metrics such as Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index are used to compare the performance of these approaches.

### Preprocessing Methods
- **Normalization:** Scales features to a range of [0, 1].
- **Transformation:** Applies logarithmic or other transformations to features.
- **PCA (Principal Component Analysis):** Reduces dimensionality while preserving variance.
- **Combination Preprocessing:** Combines methods like Transformation + Normalization (T+N) and Transformation + Normalization + PCA (T+N+PCA).

### Clustering Algorithms
1. **K-Means Clustering:**
   - Evaluated with different cluster sizes (e.g., 3, 4, 5).
2. **Hierarchical Clustering:**
   - Applied with various linkage criteria (e.g., single, complete, average).
3. **Mean Shift Clustering:**
   - Explored with varying bandwidths.

### Evaluation Metrics
- **Silhouette Score:** Measures cluster cohesion and separation.
- **Calinski-Harabasz Index:** Evaluates between-cluster dispersion relative to within-cluster dispersion.
- **Davies-Bouldin Index:** Assesses cluster compactness and separation (lower values are better).

## Results
### Performance Metrics Summary
| Algorithm       | Preprocessing | Clusters | Silhouette Score | Calinski-Harabasz | Davies-Bouldin |
|-----------------|---------------|----------|------------------|-------------------|----------------|
| K-Means         | PCA           | 3        | 0.56             | 1425.87           | 0.56          |
| Hierarchical    | PCA           | 4        | 0.48             | 1123.34           | 0.68          |
| Mean Shift      | None          | N/A      | N/A              | N/A               | N/A           |

### Key Observations
1. K-Means with PCA preprocessing achieved the highest Calinski-Harabasz scores, indicating well-separated clusters.
2. Normalization improved the Davies-Bouldin Index across most algorithms, reflecting better cluster compactness.
3. Hierarchical Clustering performed consistently with minimal preprocessing.
4. Mean Shift clustering, while consistent, required more computational resources for larger datasets.

### Visualizations
- **Cluster Performance Comparison:** Bar plots illustrating metric differences across configurations.
- **Silhouette Scores by Configuration:** Graphical representation of cluster cohesion for different methods.

## Tools and Libraries
- **Programming Language:** Python
- **Libraries:**
  - Pandas
  - Scikit-learn
  - Matplotlib (for visualizations)

## How to Run
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/sidharthd7/Clustering.git
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook:**
   Open `clustering_analysis.ipynb` in Jupyter Notebook or any compatible IDE.

## Repository Structure
- **clustering_analysis.ipynb:** Contains the clustering implementation.
- **README.md:** Documentation file (this file).
- **sample_results_table.png:** Example of result presentation.

## Conclusion
- K-Means clustering showed strong performance with smaller cluster sizes and PCA preprocessing.
- Hierarchical clustering proved effective for larger cluster sizes and required minimal preprocessing.
- Mean Shift clustering provided consistent results but was computationally intensive.

## License
This project is distributed under the MIT License. Feel free to use, modify, and distribute it as needed.

