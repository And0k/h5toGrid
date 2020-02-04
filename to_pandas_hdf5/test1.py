# def find_nb(m):
#     n_iterations = 1
#     if m == 1:
#         return n_iterations
#     m_rem = m
#     while m_rem > 8:
#         n_iterations+=1
#         m_max_root = int(m_rem**(1/3))
#         m_max = m_max_root**3
#         m_rem -= m_max
#     if m_rem !=1:
#         n_iterations = -1
#     return n_iterations


def find_nb(m):
    """
    Using Partial Sum Formula for k^3 from 1 to some value n:
    (n*(n+1)/2)**2 = m
    Finding possible n:
    (n*(n+1)/2) = sqrt(m)
    n*(n+1) = 2*sqrt(m)
    n**2 + n - 2*sqrt(m) = 0
    Discriminant:
    d = 1 - 4*(- 2*sqrt(m)) = 1 + 8*sqrt(m)
    n = x1 = (-1 + sqrt(d))/2

    # root x2 = (-1 - sqrt(d))/2 is negative nonsense
    # if d == 0 root is 1/2 so also nonsense

    d must be >= 9 to be n>=1
    """
    discriminant = 1 + 8 * m ** (1 / 2)
    if discriminant >= 9:
        n = (discriminant ** (1 / 2) - 1) / 2
        if n == int(n):
            return n
    return -1


find_nb(1806276689766864226)
