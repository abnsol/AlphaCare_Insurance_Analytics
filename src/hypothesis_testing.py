import pandas as pd
from scipy import stats
import numpy as np

def calculate_risk_metrics(df):
    """Calculates Claim Frequency, Claim Severity, and Margin."""
    df['ClaimOccurred'] = (df['TotalClaims'] > 0).astype(int)
    df['ClaimFrequency'] = df['ClaimOccurred']
    # Claim Severity is only for policies with claims
    df['ClaimSeverity'] = df['TotalClaims'].where(df['ClaimOccurred'] == 1)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def perform_t_test(group1_data, group2_data, alpha=0.05):
    """Performs independent t-test for numerical data (e.g., Claim Severity, Margin)."""
    stat, p_value = stats.ttest_ind(group1_data.dropna(), group2_data.dropna(), equal_var=False) # Welch's t-test
    return stat, p_value, p_value < alpha

def perform_chi_squared_test(group1_claims_occurred, group1_no_claims,
                             group2_claims_occurred, group2_no_claims, alpha=0.05):
    """Performs Chi-squared test for categorical data (e.g., Claim Frequency)."""
    contingency_table = np.array([[group1_claims_occurred, group1_no_claims],
                                  [group2_claims_occurred, group2_no_claims]])
    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return chi2_stat, p_value, p_value < alpha

def conduct_hypothesis_test(df, group_col, alpha=0.05):
    """
    Conducts all specified hypothesis tests for a given grouping column.
    Expects df to have 'ClaimFrequency', 'ClaimSeverity', 'Margin' calculated.
    """
    results = {}

    # Get unique groups
    groups = df[group_col].unique()
    if len(groups) < 2:
        print(f"Not enough groups in '{group_col}' to perform tests.")
        return results

    # For simplicity, pick the top two most frequent groups or specific ones if known
    # For Province/Zipcode/Gender, you might compare specific pairs or use ANOVA/Kruskal-Wallis
    # For this guide, we'll demonstrate a pairwise comparison logic.
    # For multiple groups, statistical approaches like ANOVA or Kruskal-Wallis might be better.
    # However, for the prompt's simplicity, it asks for 'no risk differences between X and Y', implying pairwise.
    # Let's generalize for two specific groups for clarity.

    # Example: Picking two groups. For "Provinces", you'd loop or pick specific top-N.
    # Or, if comparing "Women vs Men", you'd pick 'Female' and 'Male'.
    group_names = df[group_col].value_counts().index.tolist()
    if len(group_names) < 2:
        print(f"Insufficient groups for {group_col} comparison.")
        return results

    # We'll just compare the first two identified groups for demonstration
    g1_name, g2_name = group_names[0], group_names[1]
    group1_df = df[df[group_col] == g1_name]
    group2_df = df[df[group_col] == g2_name]

    # 1. Risk Differences (Claim Frequency - using Chi-squared)
    g1_claims = group1_df['ClaimOccurred'].sum()
    g1_no_claims = len(group1_df) - g1_claims
    g2_claims = group2_df['ClaimOccurred'].sum()
    g2_no_claims = len(group2_df) - g2_claims

    chi2_freq, p_freq, reject_freq = perform_chi_squared_test(g1_claims, g1_no_claims, g2_claims, g2_no_claims, alpha)
    results[f'Claim_Frequency_{g1_name}_vs_{g2_name}'] = {
        'p_value': p_freq,
        'reject_null': reject_freq,
        'interpretation': f"Claim frequency difference between {g1_name} and {g2_name} is {'significant' if reject_freq else 'not significant'}"
    }

    # 2. Risk Differences (Claim Severity - using t-test)
    stat_sev, p_sev, reject_sev = perform_t_test(group1_df['ClaimSeverity'], group2_df['ClaimSeverity'], alpha)
    results[f'Claim_Severity_{g1_name}_vs_{g2_name}'] = {
        'p_value': p_sev,
        'reject_null': reject_sev,
        'interpretation': f"Claim severity difference between {g1_name} and {g2_name} is {'significant' if reject_sev else 'not significant'}"
    }

    # 3. Margin Differences (using t-test)
    stat_margin, p_margin, reject_margin = perform_t_test(group1_df['Margin'], group2_df['Margin'], alpha)
    results[f'Margin_Difference_{g1_name}_vs_{g2_name}'] = {
        'p_value': p_margin,
        'reject_null': reject_margin,
        'interpretation': f"Margin difference between {g1_name} and {g2_name} is {'significant' if reject_margin else 'not significant'}"
    }

    return results

# Helper for specific tests in notebook
def test_province_risk_differences(df, alpha=0.05):
    print("\n--- Testing Risk Differences Across Provinces ---")
    province_results = {}
    # Pick top 2 provinces for pairwise comparison example
    top_provinces = df['Province'].value_counts().index.tolist()
    if len(top_provinces) >= 2:
        p1, p2 = top_provinces[0], top_provinces[1]
        print(f"Comparing {p1} vs {p2}...")
        province_results.update(conduct_hypothesis_test(df[df['Province'].isin([p1, p2])], 'Province', alpha))
    else:
        print("Not enough unique provinces for comparison.")
    return province_results

def test_zipcode_risk_differences(df, alpha=0.05):
    print("\n--- Testing Risk Differences Between Zipcodes ---")
    zipcode_results = {}
    # This will be very granular. Better to pick top N or a sample.
    # For simplicity of demonstration, let's just pick two prevalent ones.
    top_zipcodes = df['PostalCode'].value_counts().head(2).index.tolist()
    if len(top_zipcodes) >= 2:
        z1, z2 = top_zipcodes[0], top_zipcodes[1]
        print(f"Comparing {z1} vs {z2}...")
        zipcode_results.update(conduct_hypothesis_test(df[df['PostalCode'].isin([z1, z2])], 'PostalCode', alpha))
    else:
        print("Not enough unique zipcodes for comparison.")
    return zipcode_results

def test_zipcode_margin_differences(df, alpha=0.05):
    print("\n--- Testing Margin Differences Between Zipcodes ---")
    margin_results = {}
    top_zipcodes = df['PostalCode'].value_counts().head(2).index.tolist()
    if len(top_zipcodes) >= 2:
        z1, z2 = top_zipcodes[0], top_zipcodes[1]
        print(f"Comparing Margin for {z1} vs {z2}...")
        g1_margin = df[df['PostalCode'] == z1]['Margin']
        g2_margin = df[df['PostalCode'] == z2]['Margin']
        stat_margin, p_margin, reject_margin = perform_t_test(g1_margin, g2_margin, alpha)
        margin_results[f'Margin_Difference_{z1}_vs_{z2}'] = {
            'p_value': p_margin,
            'reject_null': reject_margin,
            'interpretation': f"Margin difference between {z1} and {z2} is {'significant' if reject_margin else 'not significant'}"
        }
    else:
        print("Not enough unique zipcodes for margin comparison.")
    return margin_results

def test_gender_risk_differences(df, alpha=0.05):
    print("\n--- Testing Risk Differences Between Women and Men ---")
    gender_results = {}
    # Ensure 'Female' and 'Male' exist
    if 'Female' in df['Gender'].unique() and 'Male' in df['Gender'].unique():
        print("Comparing Female vs Male...")
        gender_results.update(conduct_hypothesis_test(df[df['Gender'].isin(['Female', 'Male'])], 'Gender', alpha))
    else:
        print("Gender categories 'Female'/'Male' not found or insufficient for comparison.")
    return gender_results