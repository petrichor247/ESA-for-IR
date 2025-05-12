from scipy.stats import ttest_ind

def hypothesis_test(sample1, sample2):
    """
    Perform a two-sample Welch's t-test to compare the means of two samples.

    Parameters:
    - sample1, sample2: array-like, input samples

    Returns:
    - t_value: float, computed t-statistic
    - p_value: float, two-tailed p-value
    """
    t_value, p_value = ttest_ind(sample1, sample2, equal_var=False, alternative='greater')
    return t_value, p_value
