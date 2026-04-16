import numpy as np
import matplotlib.pyplot as plt

def reward_function(d, std):
    return 1 - np.tanh(d / std)

# 多组 std 对比
std_values = [0.1]
d = np.linspace(0, 1.0, 500)

plt.figure(figsize=(8, 5))
for std in std_values:
    r = reward_function(d, std)
    plt.plot(d, r, label=f'std = {std}')

plt.title(r'Reward Function: $r(d) = 1 - \tanh\left(\frac{d}{\sigma}\right)$')
plt.xlabel('Distance $d$')
plt.ylabel('Reward $r(d)$')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
