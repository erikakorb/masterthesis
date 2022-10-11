from scipy.stats import maxwell
# find median of maxwellian given the sigma
median = maxwell.median(scale=70)

print(median)
print(maxwell.median(scale=51))
print(maxwell.median(scale=265))
