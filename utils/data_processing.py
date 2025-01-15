# 📂 utils/data_processing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Function to load the dataset
def load_dataset(file_path):
    return pd.read_csv(file_path)

# Function to save the processed dataset
def save_dataset(df, output_path):
    df.to_csv(output_path, index=False)

# Function to plot correlation heatmap
def plot_correlation_heatmap(df, output_file):
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.savefig(output_file)
    plt.show()

# Function to plot scatter plots
def plot_scatter_plots(df, features, output_file):
    plt.figure(figsize=(18, 5))
    for i, feature in enumerate(features):
        plt.subplot(1, len(features), i+1)
        plt.scatter(df[feature], df['price'], alpha=0.5)
        plt.title(f'{feature} vs Price')
        plt.xlabel(feature)
        plt.ylabel('Price')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

# Function to analyze the relationship between ram and price_2
def analyze_price_transformation(merged_data):
    X = merged_data['ram'].values.reshape(-1, 1)
    y = merged_data['price_2'].values

    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Get the equation of the line
    slope = model.coef_[0]
    intercept = model.intercept_

    # Print and plot the linear relationship
    print(f"Linear Transformation: price_2 = {slope:.4f} * ram + {intercept:.2f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(merged_data['ram'], merged_data['price_2'], alpha=0.5, color='purple')
    plt.plot(merged_data['ram'], model.predict(X), color='red', label=f'price_2 = {slope:.4f} * ram + {intercept:.2f}')
    plt.title('RAM vs Price_2 with Linear Fit')
    plt.xlabel('RAM')
    plt.ylabel('Price_2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to add ordinal and nominal features
def process_features(df):
    ordinal_features = ['sc_h', 'sc_w']
    nominal_features = ['sim', 'screen', 'wifi']

    for feature in ordinal_features:
        df[f'{feature}_ord'] = pd.Categorical(df[feature]).codes

    for feature in nominal_features:
        dummies = pd.get_dummies(df[feature], prefix=feature)
        df = pd.concat([df, dummies], axis=1)

    return df

# Main function to process the dataset and create visualizations
def main():
    # Load the datasets
    file_path_1 = 'data/mobile_price_1.csv'
    file_path_2 = 'data/mobile_price_2.csv'
    output_path = 'output/mobile_prices_converted.csv'
    heatmap_output = 'assets/correlation_heatmap.png'
    scatter_output = 'assets/scatter_plots.png'

    df1 = load_dataset(file_path_1)
    df2 = load_dataset(file_path_2)

    # Merge datasets
    merged_data = pd.merge(df1, df2, on='id')

    # Plot correlation heatmap
    plot_correlation_heatmap(df1, heatmap_output)

    # Process features
    df1 = process_features(df1)

    # Save the processed dataset
    save_dataset(df1, output_path)

    # Plot scatter plots
    plot_scatter_plots(df1, ['ram', 'px_width', 'px_height'], scatter_output)

    # Analyze the relationship between ram and price_2
    analyze_price_transformation(merged_data)

if __name__ == "__main__":
    main()
