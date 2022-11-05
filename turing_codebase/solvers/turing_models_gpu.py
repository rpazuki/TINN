# from idna import valid_contexto
import numpy as np
from numba import cuda

#  $\partial_t u = D_u (\partial_x^2 + \partial_y^2)u - \alpha u + \epsilon v$
#  $\partial_t v = D_v (\partial_x^2 + \partial_y^2)v -u + \mu v - v^3$
# @numba.jit(nopython=True)
# def FitzHugh_Nagumo_2(c, t, f_args):
#     alpha, epsilon, mu = f_args
#     u = c[0, :, :]
#     v = c[1, :, :]
#     fu = epsilon * (v - alpha * u)
#     fv = -u + mu * v - v * v * v
#     return np.stack((fu, fv))

#  $\partial_t u = D_u (\partial_x^2 + \partial_y^2)u + \mu u - u^3 - v + \sigma$
#  $\partial_t v = D_v (\partial_x^2 + \partial_y^2)v +bu - \gamma v $
# @numba.jit(nopython=True)
# def FitzHugh_Nagumo(c, t, f_args):
#     sigma, b, gamma, mu = f_args
#     u = c[0, :, :]
#     v = c[1, :, :]
#     fu = mu * u - u * u * u - v + sigma
#     fv = b * u - gamma * v
#     return np.stack((fu, fv))


# def FitzHugh_Nagumo_steady_state(sigma, b, gamma, mu, init_u=1.0, it=100):
#     def F(u):
#         return u * u * u - (mu - (b / gamma)) * u - sigma

#     def F_prime(u):
#         return 3.0 * u * u - (mu - (b / gamma))

#     u_star = init_u
#     for i in range(it):
#         u_star = u_star - F(u_star) / F_prime(u_star)
#     return (u_star, (b / gamma) * u_star)


# $\partial_t u = D_u (\partial_x^2 + \partial_y^2)u + c_1 -c_0 u + c_3u^2v$
# $\partial_t v = D_v (\partial_x^2 + \partial_y^2)v + c_2 -c_3 u^2 v$
# @numba.jit(nopython=True)
# def Schnakenberg(c, t, f_args):
#     c_0, c_1, c_2, c_3 = f_args
#     u = c[0, :, :]
#     v = c[1, :, :]
#     u2v = (u**2) * v
#     fu = c_1 - c_0 * u + c_3 * u2v
#     fv = c_2 - c_3 * u2v
#     return np.stack((fu, fv))


# # $\partial_t u = D_a (\partial_x^2 + \partial_y^2)u + \rho_u \frac{u^2 v}{1 + \kappa_u u^2} - \mu_u u + \sigma_u$
# # $\partial_t v = D_s (\partial_x^2 + \partial_y^2)v - \rho_v\frac{u^2 v}{1 + \kappa_u u^2} + \sigma_v$
@cuda.jit(nopython=True)
def Koch_Meinhardt(c, f_args, z):
    kappa_u, mu_u, rho_u, rho_v, sigma_u, sigma_v   = f_args
    u = c[0]
    v = c[1]
    u2 = u**2
    u2v = u2 * v
    u2v_u2 = u2v / (1.0 + kappa_u * u2)
    if z == 0:
        return rho_u * u2v_u2 - mu_u * u + sigma_u
    elif z==1:
        return -rho_v * u2v_u2 + sigma_v
    else:
        assert "Wrong number of species"


#  $\partial_t u = D_u (\partial_x^2 + \partial_y^2)u + A - (B+1)u + u^2v$
#  $\partial_t v = D_v (\partial_x^2 + \partial_y^2)v + Bu - u^2 v$
@cuda.jit
def Brusselator(c, f_args, z):
    A, B = f_args
    u = c[0]
    v = c[1]
    u2 = u**2
    u2v = u2 * v
    if z == 0:
        return A - (B + 1) * u + u2v
    elif z==1:
        return B * u - u2v
    else:
        assert "Wrong number of species"


@cuda.jit
def Circuit_3954(c, f_args, z):
    (b_A, b_B, b_C, b_D, b_E, b_F,
    n_aTc,
    K_AB, K_BD, K_CE, K_DA, K_EB, K_EE, K_FE, K_aTc,
    μ_U, μ_V, μ_B, μ_C, μ_D, μ_E, μ_F, μ_aTc) =  f_args

    U = c[0, :, :]
    V = c[1, :, :]
    A = c[2, :, :]
    B = c[3, :, :]
    C = c[4, :, :]
    D = c[5, :, :]
    E = c[6, :, :]
    F = c[7, :, :]
    aTc = c[8, :, :]

    def activate(Concentration, K, power=3):
            act = 1 / (1 + (K / (Concentration + 1e-20)) ** power)
            return act

    def inhibit(Concentration, K, power=3):
        inh = 1 / (1 + (Concentration / (K + 1e-20)) ** power)
        return inh


    if z == 0:
        return A - μ_U * U
    elif z == 1:
        return   B - μ_V * V
    elif z == 2:
        return b_A**2 + b_A * inhibit(D, K_DA) - A
    elif z == 3:
        return μ_B * (b_B**2 + b_B * activate(U, K_AB) * inhibit(E, K_EB) - B)
    elif z == 4:
        return μ_C * (b_C**2 + b_C * inhibit(D, K_DA) - C)
    elif z == 5:
        return μ_D * (b_D**2 + b_D * activate(V, K_BD) - D)
    elif z == 6:
        K_CE_star = K_CE * inhibit(aTc, K_aTc, n_aTc)
        return μ_E * (b_E**2 + b_E * inhibit(C, K_CE_star) * inhibit(F, K_FE) * activate(E, K_EE) - E)
    elif z == 7:
        return μ_F * (b_F**2 + b_F * activate(V, K_BD) - F)
    elif z == 8:
        return -μ_aTc * aTc
    else:
        assert "Wrong number of species"