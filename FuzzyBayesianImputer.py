import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances

class FuzzyBayesianImputer:
    def __init__(self, alpha=1.0, n_iter=10, tol=1e-4):
        self.alpha = alpha
        self.n_iter = n_iter
        self.tol = tol
        self.variances_ = None
        self.means_ = None
        
    def _matrix_factor(self, X, missing_mask):
        """
        Calculate fuzzy similarity between instances
        """
        # Standardize the data
        X_std = (X - self.means_) / np.sqrt(self.variances_)
        
        # Calculate Euclidean distances
        distances = euclidean_distances(X_std)
        
        # Convert distances to similarities (fuzzy membership values)
        max_dist = np.max(distances[distances != 0])
        similarities = 1 - (distances / (max_dist + 1e-10))
        
        # Apply fuzzy exponent
        similarities = similarities ** self.alpha
        
        # Set self-similarity to 0
        np.fill_diagonal(similarities, 0)
        
        # Adjust similarities for missing values
        for i in range(X.shape[0]):
            if missing_mask[i].any():
                # Reduce influence of instances that are missing the same features
                common_missing = missing_mask[i] & missing_mask
                similarities[i, common_missing.any(axis=1)] *= 0.5
        
        return similarities
    
    def _bayesian_update(self, X, missing_mask, similarities):
        """
        Perform Bayesian update for missing values
        """
        X_imputed = X.copy()
        
        for i in range(X.shape[0]):
            missing_cols = np.where(missing_mask[i])[0]
            
            for col in missing_cols:
                # Get observed values for this feature from other instances
                observed_mask = ~missing_mask[:, col]
                observed_values = X[observed_mask, col]
                
                if len(observed_values) == 0:
                    # If no observed values, use mean
                    X_imputed[i, col] = self.means_[col]
                    continue
                
                # Calculate weighted prior (from similar instances)
                weights = similarities[i, observed_mask]
                weights = weights / (weights.sum() + 1e-10)
                
                prior_mean = np.dot(weights, observed_values)
                prior_var = self.variances_[col]
                
                # Bayesian update (assuming normal distribution)
                # Here we use a simple approximation
                posterior_mean = prior_mean
                X_imputed[i, col] = posterior_mean
        
        return X_imputed
    
    def fit(self, X):
        """
        Learn the mean and variance of each feature
        """
        X = np.array(X)
        self.means_ = np.nanmean(X, axis=0)
        self.variances_ = np.nanvar(X, axis=0)
        return self
    
    def transform(self, X):
        """
        Perform fuzzy Bayesian imputation
        """
        X = np.array(X)
        missing_mask = np.isnan(X)
        X_imputed = X.copy()
        
        # Initialize with mean imputation
        for col in range(X.shape[1]):
            col_mean = self.means_[col]
            X_imputed[missing_mask[:, col], col] = col_mean
        
        # Iterative Bayesian updating
        for _ in range(self.n_iter):
            similarities = self._matrix_factor(X_imputed, missing_mask)
            new_imputed = self._bayesian_update(X_imputed, missing_mask, similarities)
            
            # Check convergence
            diff = np.linalg.norm(new_imputed[missing_mask] - X_imputed[missing_mask])
            if diff < self.tol:
                break
                
            X_imputed = new_imputed
        
        return X_imputed
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        data = self.fit(X).transform(X)
        return data

    







    
