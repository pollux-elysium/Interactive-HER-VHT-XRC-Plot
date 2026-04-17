import numpy as np

#params : [f, BDFE, log k1f, lambda1, beta1, log k2f, lambda2, beta2, log kt, lambdaT, log kh1f, lambdah1, betah1, log kh2f, lambdah2, betah2]
# 0:f
# 1:BDFE
# 2:log k1f
# 3:lambda1
# 4:beta1
# 5:log k2f
# 6:lambda2
# 7:beta2
# 8:log kt
# 9:lambdaT
#10:log kh1f
#11:lambdah1
#12:betah1
#13:log kh2f
#14:lambdah2
#15:betah2

F = 96485.33212
R = 8.31446261815324
T = 298.15
Kw = 1e-14  # water ionization constant

def equilibrium_constants(BDFE):
    T = 293.15
    dGv = 221-100*BDFE
    Kv = np.exp(-1000*dGv/(R*T))
    dGh = 221 - 442 + 100*BDFE
    Kh = np.exp(-1000*dGh/(R*T))
    dGt = -442 + 200*BDFE
    Kt = np.exp(-1000*dGt/(R*T))

    #
    # print(f"Equilibrium Constants: Kv={Kv}, Kh={Kh}, Kt={Kt}")
    return Kv, Kh, Kt

def heyrovsky1forward(params,E,pH,theta):
    f = params[0]
    BDFE = params[1]
    _, Kh, _ = equilibrium_constants(BDFE)
    E0h = R*T*np.log(Kh)/F

    kh1f = 10**(params[10])
    lambdah1 = params[11]
    betah1 = params[12]

    v = kh1f * np.power(10,-pH) * theta * np.exp( (1-lambdah1) * f * theta) * np.exp((-betah1) * F * (E - E0h)/ (R * T))
    return v

def heyrovsky1reverse(params,E,theta):
    f = params[0]
    BDFE = params[1]
    _, Kh, _ = equilibrium_constants(BDFE)
    E0h = R*T*np.log(Kh)/F

    kh1f = 10**(params[10])
    lambdah1 = params[11]
    betah1 = params[12]
    kh1r = kh1f / Kh

#                   H2 solubility    water concentration
    v = kh1r * 0.000793650794 * 55.5 * (1 - theta) * np.exp( lambdah1 * f * theta) * np.exp((1 - betah1) * F * (E - E0h)/ (R * T))
    return v

def heyrovsky2forward(params,E,theta):
    f = params[0]
    BDFE = params[1]
    _, Kh, _ = equilibrium_constants(BDFE)
    E0h = R*T*np.log(Kh)/F

    kh2f = 10**(params[13])
    lambdah2 = params[14]
    betah2 = params[15]

    v = kh2f * 55.5 * theta * np.exp( (1-lambdah2) * f * theta) * np.exp((-betah2) * F * (E - E0h + 0.82818766347)/ (R * T))
    return v
    
def heyrovsky2reverse(params,E,pH,theta):
    f = params[0]
    BDFE = params[1]
    _, Kh, _ = equilibrium_constants(BDFE)
    E0h = R*T*np.log(Kh)/F

    kh2f = 10**(params[13])
    lambdah2 = params[14]
    betah2 = params[15]
    kh2r = kh2f / Kh

    v = kh2r * 0.000793650794 * (1-theta) * Kw * np.power(10,pH) * np.exp( lambdah2 * f * theta) * np.exp((1 - betah2) * F * (E - E0h + 0.82818766347)/ (R * T))
    return v

def tafelforward(params,theta):
    f = params[0]
    BDFE = params[1]

    ktf = 10**(params[8])
    lambdaT = params[9]

    v = ktf * theta**2 * np.exp( 2*(1 - lambdaT) * f * theta)
    return v

def tafelreverse(params,theta):
    f = params[0]
    BDFE = params[1]
    _,_,Kt = equilibrium_constants(BDFE)

    ktf = 10**(params[8])
    ktr = ktf / Kt
    lambdaT = params[9]

    v = ktr * 0.000793650794* (1 - theta)**2 * np.exp( 2*lambdaT * f * theta)
    return v

def volmer1forward(params,E,pH,theta):
    f = params[0]
    BDFE = params[1]
    Kv, _, _ = equilibrium_constants(BDFE)
    E0v = R*T*np.log(Kv)/F
    kv1f = 10**(params[2])
    lambda1 = params[3]
    beta1 = params[4]

    v = kv1f *(1-theta)*np.power(10,-pH) * np.exp(-lambda1 * f * theta) * np.exp((-beta1) * F * (E - E0v)/ (R * T))
    return v

def volmer1reverse(params,E,theta):
    f = params[0]
    BDFE = params[1]
    Kv, _, _ = equilibrium_constants(BDFE)
    E0v = R*T*np.log(Kv)/F
    kv1f = 10**(params[2])
    lambda1 = params[3]
    beta1 = params[4]
    kv1r = kv1f / Kv

    v = kv1r * 55.5 * theta * np.exp((1-lambda1) * f * theta) * np.exp((1 - beta1) * F * (E - E0v)/ (R * T))
    return v

def volmer2forward(params,E,theta):
    f = params[0]
    BDFE = params[1]
    Kv, _, _ = equilibrium_constants(BDFE)
    E0v = R*T*np.log(Kv)/F
    kv2f = 10**(params[5])
    lambda2 = params[6]
    beta2 = params[7]

    v = kv2f * 55.5 * (1-theta) * np.exp(-lambda2 * f * theta) * np.exp((-beta2) * F * (E - E0v + 0.82818766347)/ (R * T))
    return v

def volmer2reverse(params,E,pH,theta):
    f = params[0]
    BDFE = params[1]
    Kv, _, _ = equilibrium_constants(BDFE)
    E0v = R*T*np.log(Kv)/F
    kv2f = 10**(params[5])
    lambda2 = params[6]
    beta2 = params[7]
    kv2r = kv2f / Kv

    v = kv2r * theta * Kw * np.power(10,pH) * np.exp((1 - lambda2) * f * theta) * np.exp((1 - beta2) * F * (E - E0v + 0.82818766347)/ (R * T))
    return v
