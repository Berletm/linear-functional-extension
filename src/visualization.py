import numpy as np
import matplotlib.pyplot as plt

g_vec = np.array([3, 5, 4, 6], dtype=np.float32)
k_vec = np.array([1, 2, 9, 2], dtype=np.float32)

a = np.array([37, -23, 1, 0], dtype=np.float32)
a /= np.linalg.norm(a)

b = np.array([-2, 0, 0, 1], dtype=np.float32)
b = b - np.dot(b, a) * a
b /= np.linalg.norm(b)

c_raw = np.array([0.5, 0.784689, -0.452153, 1.0], dtype=np.float32)
c = c_raw - np.dot(c_raw, a)*a - np.dot(c_raw, b)*b
c /= np.linalg.norm(c)

ga = np.dot(g_vec, a)
gb = np.dot(g_vec, b)
gc = np.dot(g_vec, c)
g_in_basis = np.array([ga, gb, gc])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

origin = [0, 0, 0]
ax.quiver(*origin, 1, 0, 0, color='r', label='Базис a (в ядре)', arrow_length_ratio=0.1)
ax.quiver(*origin, 0, 1, 0, color='g', label='Базис b (в ядре)', arrow_length_ratio=0.1)
ax.quiver(*origin, 0, 0, 1, color='b', label='Базис c (вне ядра)', arrow_length_ratio=0.1)

xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
zz = np.zeros_like(xx)
ax.plot_surface(xx, yy, zz, alpha=0.2, color='dimgray')

ax.quiver(*origin, ga, gb, gc, color='purple', label='Вектор g', linewidth=3, arrow_length_ratio=0.05)

max_val = max(abs(ga), abs(gb), abs(gc), 1.5) * 0.8 # +20% запаса

limit = np.max(np.abs(g_in_basis)) * 0.6

ax.set_xlim([-limit, limit])
ax.set_ylim([-limit, limit])
ax.set_zlim([-limit, limit])

ax.set_box_aspect([1,1,1])
ax.dist = 11 


ax.set_xlabel('Ось a')
ax.set_ylabel('Ось b')
ax.set_zlabel('Ось c')
ax.set_title('Визуализация функционала g в подпространстве Y')
ax.legend()

plt.tight_layout()
plt.savefig("../report/utils/figure.png", dpi=500)