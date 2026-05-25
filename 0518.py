import numpy as np

# --------------------------
# 单元刚度矩阵计算
# --------------------------
def truss3d_element_stiffness(x1, x2, E, A):

    # ==================== 单元长度计算 =====================
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx**2 + dy**2 + dz**2)

    # ===================== 方向余弦 =======================

    cx, cy, cz = dx/L, dy/L, dz/L

    T = np.array([[cx, cy, cz, 0, 0, 0],
                  [0, 0, 0, cx, cy, cz]])

    k_local = (E * A / L) * np.array([[1, -1], [-1, 1]])
    Ke = T.T @ k_local @ T

    return L, (cx, cy, cz), Ke


# ============================ 应变/应力/轴力计算 ==========================
def truss3d_element_stress(x1, x2, E, A, de):

    L, (cx, cy, cz), _ = truss3d_element_stiffness(x1, x2, E, A)
    B = np.array([-cx, -cy, -cz, cx, cy, cz]) / L

    epsilon = B @ de
    sigma = E * epsilon
    N = sigma * A
    return epsilon, sigma, N

# --------------------------
# 辅助：验证刚度矩阵性质
# --------------------------
def check_Ke_properties(Ke, tol=1e-8):
    sym_ok = np.allclose(Ke, Ke.T, atol=tol)
    rank = np.linalg.matrix_rank(Ke, tol=tol)
    eigvals = np.linalg.eigvalsh(Ke)
    pos_semi = np.all(eigvals >= -tol)
    singular = rank < 6
    return sym_ok, rank, eigvals, pos_semi, singular

# --------------------------
# 任务3：验证算例
# --------------------------
if __name__ == "__main__":
    # 单位统一：N, m, Pa
    print("="*60)
    print("算例1：沿x轴一维杆单元")
    print("="*60)
    x1 = [0.0, 0.0, 0.0]
    x2 = [2.0, 0.0, 0.0]
    E = 200e9
    A = 1.0e-4
    de = np.array([0, 0, 0, 1.0e-3, 0, 0])

    L, dir_cos, Ke = truss3d_element_stiffness(x1, x2, E, A)
    eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)

    print(f"单元长度 L = {L:.4f} m")
    print(f"方向余弦 (cx,cy,cz) = {dir_cos}")
    print("刚度矩阵 Ke (6x6):")
    print(np.array2string(Ke, precision=4, suppress_small=True))
    print(f"应变 ε = {eps:.6f}")
    print(f"应力 σ = {sig/1e6:.2f} MPa")
    print(f"轴力 N = {N:.2f} N")

    print("\n" + "="*60)
    print("算例2：空间任意方向杆单元")
    print("="*60)
    x1 = [0.0, 0.0, 0.0]
    x2 = [1.0, 2.0, 2.0]
    E = 210e9
    A = 2.0e-4
    de = np.array([0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3])

    L, dir_cos, Ke = truss3d_element_stiffness(x1, x2, E, A)
    eps, sig, N = truss3d_element_stress(x1, x2, E, A, de)
    sym_ok, rank, eigvals, pos_semi, singular = check_Ke_properties(Ke)

    print(f"单元长度 L = {L:.4f} m")
    print(f"方向余弦 (cx,cy,cz) = {dir_cos}")
    print("刚度矩阵 Ke (6x6):")
    print(np.array2string(Ke, precision=4, suppress_small=True))
    print(f"应变 ε = {eps:.6f}")
    print(f"应力 σ = {sig/1e6:.2f} MPa")
    print(f"轴力 N = {N:.2f} N")
    print("\n刚度矩阵性质：")
    print(f"对称性: {sym_ok}")
    print(f"秩: {rank} (奇异: {singular})")
    print(f"半正定性: {pos_semi}")
    print(f"特征值: {np.round(eigvals, 4)}")

    print("\n" + "="*60)
    print("任务4：刚度矩阵物理意义验证（第4列）")
    print("="*60)
    de_j = np.zeros(6)
    de_j[3] = 1.0  # 第4个自由度(u2)=1，其余0
    Fe = Ke @ de_j
    print(f"节点力 Fe = {np.round(Fe, 4)}")
    print("结论：Fe 等于 Ke 的第4列，kij 表示j自由度单位位移引起的i自由度力")