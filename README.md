# Data-Normalization-and-Data-Type-Conversion
## Theory

This experiment demonstrates data normalization techniques to scale numerical features and methods to convert categorical (qualitative) variables into numerical (quantitative) variables for machine learning models.

The notebook covers the following:
1. Importing required libraries  
   ```python
   import pandas as pd
   import numpy as np
   from sklearn.preprocessing import LabelEncoder
   ```
   Pandas is used for data manipulation, NumPy for numerical operations, and LabelEncoder from scikit-learn for label encoding.

2. Creating sample product dataset and applying Min-Max Normalization  
   ```python
   df['Price_MinMax'] = (df['Price'] - df['Price'].min()) / (df['Price'].max() - df['Price'].min())
   ```  
   Min-Max Normalization scales values to range [0,1] using the formula: (x - min) / (max - min). It is applied separately on Price, Units_Sold, and Discount columns.

3. Z-Score Normalization (Standardization)  
   ```python
   df['Units_Zscore'] = (df['Units_Sold'] - df['Units_Sold'].mean()) / df['Units_Sold'].std()
   ```  
   Z-Score transforms data to have mean=0 and standard deviation=1 using: (x - mean) / std. This helps in handling features with different scales.

4. Decimal Scaling Normalization  
   ```python
   df['Price_Decimal'] = df['Price'] / 100000
   ```  
   Decimal Scaling moves the decimal point to bring values into smaller range (usually between -1 and 1) by dividing by a power of 10 based on the maximum number of digits.

5. Batch Min-Max Normalization on multiple columns  
   ```python
   cols = ['Price', 'Units_Sold', 'Discount']
   df_norm = (df[cols] - df[cols].min()) / (df[cols].max() - df[cols].min())
   ```  
   This applies Min-Max normalization to multiple columns at once using vectorized operations.

6. Loading Amazon products dataset and applying normalization  
   The notebook reads an external CSV file and applies Min-Max, Z-Score, and Decimal Scaling on columns like Price, Rating, Reviews, and Units_Sold.

7. Label Encoding for categorical variables  
   ```python
   le = LabelEncoder()
   df['Gender_Label'] = le.fit_transform(df['Customer_Gender'])
   ```  
   Label Encoding converts categorical values into numeric labels (e.g., Male=1, Female=0). It is suitable for ordinal data or tree-based models.

8. One-Hot Encoding and Dummy Encoding  
   ```python
   df_encoded = pd.get_dummies(df, columns=['Payment_Method'])
   df_dummy = pd.get_dummies(df, columns=['Payment_Method'], drop_first=True)
   ```  
   One-Hot Encoding creates binary columns for each category. Dummy Encoding drops the first category to avoid multicollinearity (dummy variable trap).

These techniques help in:
- Scaling numerical features to similar ranges (Normalization)
- Converting text categories into numbers for model compatibility (Encoding)

## Conclusion

The experiment successfully demonstrated how to:
- Perform Min-Max, Z-Score, and Decimal Scaling normalization on numerical data
- Convert categorical variables to numerical using Label Encoding, One-Hot Encoding, and Dummy Encoding
- Apply these techniques on both custom and real-world datasets (Amazon products and Student data)

These preprocessing steps are essential for improving model performance, ensuring fair feature contribution, and making data suitable for machine learning algorithms.
