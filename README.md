# Tehran House Price Prediction & Analysis

This project builds a machine learning model to predict house prices in Tehran. Beyond just prediction, the notebook provides an in-depth analysis of the model's performance, feature importance, and a deep dive into the sources of prediction errors using model interpretation techniques like LIME.

## Project Workflow

1.  **Data Loading and Cleaning:**
    *   The dataset `data/housePrice.csv` is loaded.
    *   The original 'Price' column (in Toman) is dropped in favor of 'Price(USD)'.
    *   Rows with missing `Address` values are removed.
    *   The `Area` column is converted to a numeric type, and any resulting nulls are dropped.

2.  **Feature Engineering:**
    *   `Area_per_Room` is created to normalize the area by the number of rooms.
    *   A custom `Area_Ranking` feature is engineered, assigning a prestige score (1-5) to each neighborhood based on a predefined dictionary. This injects domain knowledge into the model.

3.  **Outlier Removal:**
    *   To prevent extreme values from skewing the model, the top 5% and bottom 1% of properties based on price are removed from the dataset.

4.  **Preprocessing & Model Training:**
    *   Categorical features (`Room`, `Parking`, `Warehouse`, `Elevator`) are converted to the correct data type.
    *   The `Address` column is one-hot encoded, creating a wide feature set.
    *   Features and the target variable (`Price`) are scaled using `MinMaxScaler`.
    *   Several regression models are evaluated using 10-fold cross-validation, including Ridge, Lasso, Decision Tree, K-Neighbors, AdaBoost, and Random Forest.

5.  **Model Selection and Error Analysis:**
    *   **Random Forest Regressor** is selected as the best-performing model based on the lowest Root Mean Squared Error (RMSE).
    *   A detailed error analysis is performed on the test set to identify patterns in mispredictions, focusing on cases with a >70% prediction error.
    *   Feature importance is calculated using both the standard (Gini) importance from Random Forest and the more robust Grouped Permutation Importance.
    *   **LIME (Local Interpretable Model-agnostic Explanations)** is used to explain the top 10 worst predictions, revealing which features contributed most to the errors on a case-by-case basis.

## Key Findings

*   **Best Model:** `RandomForestRegressor` achieved the best performance with a Mean RMSE of **$52,951** on the cross-validation set.
*   **Most Important Features:** Numerical features (`Area`, `Area_per_Room`) and the engineered `Area_Ranking` are the most significant drivers of price. The one-hot encoded `Address` features, when grouped, are the next most important category.
*   **Error Patterns:** The model struggles most with very low-priced properties, often over-predicting their value significantly.
*   **Problematic Locations:** The analysis revealed that a few specific addresses (e.g., 'Thirteen November', 'Nasim Shahr') had extremely high error rates, likely due to having very few data points in the dataset, making it difficult for the model to learn their price patterns.

## How to Run

1.  **Prerequisites:** Ensure you have Python 3.x installed.

2.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

3.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib scikit-learn lime
    ```

4.  **Data:** Make sure the `data/housePrice.csv` file is present in the specified directory.

5.  **Run the Notebook:** Launch Jupyter and run the cells in the `house_price_prediction.ipynb` notebook sequentially.