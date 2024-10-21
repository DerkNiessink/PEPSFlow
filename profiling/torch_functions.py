import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")
from profiling import profile, percentage_cpu_time


chi_values = [chi for chi in range(5, 50, 2)]
profs = [profile(gpu=False, chi=chi, d = 2, max_iter = 1, epochs=1) for chi in chi_values]
print(profs[-1].key_averages().table(sort_by='cpu_time_total', row_limit=10))

percentages_reshape = [percentage_cpu_time(prof, 'aten::reshape') for prof in profs]
percentages_svd = [percentage_cpu_time(prof, 'aten::linalg_svd') for prof in profs]
percentages_einsum = [percentage_cpu_time(prof, 'aten::einsum') for prof in profs]
percentages_permute = [percentage_cpu_time(prof, 'aten::permute') for prof in profs]

plt.figure(figsize=(6, 4))
plt.plot(chi_values, percentages_svd, 'o-', label='linalg_svd', markersize=3, linewidth=0.5)
plt.plot(chi_values, percentages_einsum, 'o-', label='einsum', markersize=3, linewidth=0.5)
plt.plot(chi_values, percentages_permute, 'o-', label='permute', markersize=3, linewidth=0.5)
plt.plot(chi_values, percentages_reshape, 'o-', label='reshape', markersize=3, linewidth=0.5)
plt.xlabel(r'$\chi$', fontsize=14)
plt.ylabel(r'Total CPU \%', fontsize=14)
plt.legend()
plt.show()