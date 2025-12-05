# Wine Classification Project

## Overview
Multi-class classification of wine types using chemical properties. Achieved 95%+ accuracy with optimized feature selection.

## Key Findings
- Random Forest outperformed Decision Tree Classifier
- Only 5 of 13 features needed for 95%+ accuracy  
- Cross-validation revealed overfitting in initial models

## Technologies
- Python, scikit-learn, pandas, numpy, matplotlib
- Random Forest, Decision Tree, SVM Classifier
- Feature selection, cross-validation

## Dataset
- Source: UCI Machine Learning Repository via scikit-learn
- Load with: `from sklearn.datasets import load_wine`
- 178 samples, 13 features, 3 wine classes
- No external data files required