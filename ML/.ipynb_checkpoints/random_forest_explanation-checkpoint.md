# Random Forest Classification and Regression

## Overview
Random Forest is an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. It uses:
- **Bagging**: Bootstrap sampling of training data
- **Random Feature Selection**: Subset of features at each split
- **Voting/Averaging**: Combines predictions from all trees

## Random Forest Architecture

```mermaid
graph TD
    A[Training Dataset] --> B[Bootstrap Sample 1]
    A --> C[Bootstrap Sample 2] 
    A --> D[Bootstrap Sample 3]
    A --> E[Bootstrap Sample N]
    
    B --> F[Decision Tree 1]
    C --> G[Decision Tree 2]
    D --> H[Decision Tree 3]
    E --> I[Decision Tree N]
    
    F --> J[Prediction 1]
    G --> K[Prediction 2]
    H --> L[Prediction 3]
    I --> M[Prediction N]
    
    J --> N[Ensemble Prediction]
    K --> N
    L --> N
    M --> N
    
    style A fill:#e1f5fe
    style N fill:#c8e6c9
```

## Random Forest Classification

### Process Flow
```mermaid
flowchart TD
    A[New Sample] --> B[Tree 1: Class A]
    A --> C[Tree 2: Class B]
    A --> D[Tree 3: Class A]
    A --> E[Tree N: Class A]
    
    B --> F[Majority Voting]
    C --> F
    D --> F
    E --> F
    
    F --> G[Final Prediction: Class A]
    
    style A fill:#fff3e0
    style G fill:#e8f5e8
```

### Classification Example
```mermaid
graph LR
    subgraph "Tree Predictions"
        T1[Tree 1: Cat 🐱]
        T2[Tree 2: Dog 🐶]
        T3[Tree 3: Cat 🐱]
        T4[Tree 4: Cat 🐱]
        T5[Tree 5: Cat 🐱]
    end
    
    T1 --> V[Voting System]
    T2 --> V
    T3 --> V
    T4 --> V
    T5 --> V
    
    V --> R[Result: Cat 🐱<br/>4 votes vs 1 vote]
    
    style V fill:#f3e5f5
    style R fill:#e8f5e8
```

## Random Forest Regression

### Process Flow
```mermaid
flowchart TD
    A[New Sample] --> B[Tree 1: 25.3]
    A --> C[Tree 2: 24.7]
    A --> D[Tree 3: 25.8]
    A --> E[Tree N: 25.1]
    
    B --> F[Average Calculation]
    C --> F
    D --> F
    E --> F
    
    F --> G[Final Prediction: 25.2]
    
    style A fill:#fff3e0
    style G fill:#e8f5e8
```

### Regression Example
```mermaid
graph TD
    subgraph "Tree Predictions"
        T1[Tree 1: $45,000]
        T2[Tree 2: $47,500]
        T3[Tree 3: $46,200]
        T4[Tree 4: $45,800]
        T5[Tree 5: $46,500]
    end
    
    T1 --> A[Average Calculator]
    T2 --> A
    T3 --> A
    T4 --> A
    T5 --> A
    
    A --> R[Result: $46,200<br/>Average of all predictions]
    
    style A fill:#f3e5f5
    style R fill:#e8f5e8
```

## Feature Selection Process

```mermaid
graph TD
    A[All Features<br/>F1, F2, F3, F4, F5, F6] --> B[Random Subset Selection]
    
    B --> C[Tree 1: F1, F3, F5]
    B --> D[Tree 2: F2, F4, F6]
    B --> E[Tree 3: F1, F2, F4]
    B --> F[Tree N: F3, F5, F6]
    
    C --> G[Decision Tree 1]
    D --> H[Decision Tree 2]
    E --> I[Decision Tree 3]
    F --> J[Decision Tree N]
    
    style A fill:#e3f2fd
    style B fill:#f3e5f5
```

## Key Differences

| Aspect | Classification | Regression |
|--------|----------------|------------|
| **Output** | Class labels | Continuous values |
| **Combination** | Majority voting | Averaging |
| **Example** | Spam/Not Spam | House price prediction |
| **Metrics** | Accuracy, Precision, Recall | MSE, RMSE, MAE |

## Advantages of Random Forest

```mermaid
mindmap
  root((Random Forest<br/>Advantages))
    Robustness
      Reduces overfitting
      Handles missing values
      Resistant to outliers
    Performance
      High accuracy
      Good generalization
      Parallel processing
    Interpretability
      Feature importance
      Out-of-bag error
      No need for pruning
    Versatility
      Classification & Regression
      Handles mixed data types
      No assumption about distribution
```

## Hyperparameters

```mermaid
graph LR
    A[Random Forest] --> B[n_estimators<br/>Number of trees]
    A --> C[max_depth<br/>Tree depth limit]
    A --> D[max_features<br/>Features per split]
    A --> E[min_samples_split<br/>Min samples to split]
    A --> F[min_samples_leaf<br/>Min samples in leaf]
    A --> G[bootstrap<br/>Sample with replacement]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#f3e5f5
    style D fill:#f3e5f5
    style E fill:#f3e5f5
    style F fill:#f3e5f5
    style G fill:#f3e5f5
```

## Implementation Example (Python)

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Regression
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Training and prediction
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
```

## When to Use Random Forest

- **Good for**: Mixed data types, missing values, feature selection
- **Avoid when**: Interpretability is crucial, very large datasets, real-time predictions
- **Best practices**: Tune hyperparameters, check feature importance, use cross-validation
