import numpy as np
import sympy as sp

g = np.array([3, 5, 4, 6], dtype=np.float32)
k = np.array([1, 2, 9, 2], dtype=np.float32)
    
a = np.array([37, -23, 1, 0], dtype=np.float32)
a = a / np.linalg.norm(a)

b = np.array([-2, 0, 0, 1], dtype=np.float32)
b = b - np.dot(b, a) * a
b = b / np.linalg.norm(b)

c = np.array((0.5, 0.784689, -0.452153, 1.0), dtype=np.float32)
c = c / np.linalg.norm(c)
d = k / np.linalg.norm(k)

def main() -> None:    
    mat = np.array([a, b, c, d], dtype=np.float32)
    Y = np.array([a, b, c])
    print(Y @ g)
    rhs = np.array([0, 0, g@c, 0], dtype=np.float32)
    f   = np.linalg.solve(mat.T, rhs)
    print(f)
    print(g@c)
    print(np.linalg.norm(g))
    print(np.linalg.norm(f))
    
    


if __name__ == "__main__":
    main()