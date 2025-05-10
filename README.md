
```markdown
# Mobile Price Prediction

This project analyzes a dataset of mobile phone specifications to predict their prices using data visualization and machine learning techniques.

## Project Overview

The dataset includes various features such as RAM, pixel dimensions, and more. The analysis involves:

- **Correlation Heatmap**: Identifying relationships between features and the target variable, `price`.
- **Scatter Plots**: Visualizing the impact of features like `ram`, `px_width`, and `px_height` on `price`.
- **Feature Engineering**: Incorporating ordinal and nominal features to enhance model performance.
- **Price Transformation**: Applying linear transformations to the `price` variable for better model fitting.

2. **Install required packages**:

   Ensure you have Python 3.x installed. Then, install the necessary libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn

## Dependencies

* Python 3.x
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## Results

Key findings include:

* **RAM** has a strong positive correlation with **price**.
* Visualizations reveal linear relationships between certain features and the target variable.
* Feature engineering and transformation techniques improved model accuracy.
