import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

plt.figure(figsize=(6, 4))
vals = [-0.6602310927203838, -0.6675904643214068, -0.6689536180423972, -0.6693704807555166, -0.6693671933541275]
x = [2, 3, 4, 5, 6]
plt.plot(x, vals, marker="o", linestyle="-", color="k", label="Energy")
plt.xlabel("$D$", fontsize=14)
plt.ylabel("$E$", fontsize=14)
plt.xticks(x, fontsize=13)
plt.yticks(fontsize=13)
plt.ylim(-0.67, -0.66)
plt.show()
