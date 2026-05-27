import numpy as np

# 递归实现 FFT
def fft_generate(func):
    n = len(func)
    if n == 1:
        return np.array([func[0]], dtype=complex)

    func_even = fft_generate(func[::2])  # 偶数项
    func_odd = fft_generate(func[1::2])  # 奇数项

    fft_result = np.zeros(n, dtype=complex)
    for i in range(n // 2):
        w = np.exp(-2j * np.pi * i / n)  # 旋转因子
        fft_result[i] = func_even[i] + w * func_odd[i]
        fft_result[i + n // 2] = func_even[i] - w * func_odd[i]
    return fft_result


# 递归实现 IFFT
def ifft_generate(func):
    n = len(func)
    if n == 1:
        return np.array([func[0]], dtype=complex)

    func_even = ifft_generate(func[::2])
    func_odd = ifft_generate(func[1::2])

    ifft_result = np.zeros(n, dtype=complex)
    for i in range(n // 2):
        w = np.exp(2j * np.pi * i / n)  # 旋转因子（正方向）
        ifft_result[i] = func_even[i] + w * func_odd[i]
        ifft_result[i + n // 2] = func_even[i] - w * func_odd[i]

    return ifft_result / 2  # 归一化


# 进行 FFT 乘法
def fft_multiply(func1, func2):
    n = 1
    while n < len(func1) + len(func2) - 1:
        n *= 2  # 取 2 的次幂

    # 补零
    func1 = np.array(func1 + [0] * (n - len(func1)), dtype=complex)
    func2 = np.array(func2 + [0] * (n - len(func2)), dtype=complex)

    fft1 = fft_generate(func1)
    fft2 = fft_generate(func2)

    fft_result = fft1 * fft2
    ifft_result = ifft_generate(fft_result)  # 进行逆FFT 还原

    return np.real(ifft_result)  # 取实部作为最终卷积结果


# 测试数据
func_test1 = [1, 2, 3, 4, 5, 6, 7, 8]
func_test2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]

print(fft_multiply(func_test1, func_test2))
