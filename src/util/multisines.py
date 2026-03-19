import numpy as np


def multisine_clip(T, N, P, f0, seed, shroeder):
    
    rng = np.random.default_rng(seed)
    
    t = np.linspace(0,T,N).reshape(N,1)
    ks = np.linspace(1,P,P).reshape(1,P)

    if shroeder:
        # Shroeder phases
        initphase = np.zeros((P,))
        initphase[0] = rng.uniform(low=0.0, high=1.0, size=(1,))
        for ii in range(1,P):
            initphase[ii] = initphase[0] - np.pi*ii*(ii-1)/P
    else:
        initphase = 2*np.pi*rng.uniform(low=0.0, high=1.0, size=(P,))
    Cinit = (1/np.sqrt(P/2))*np.cos(2*np.pi*ks*t*f0/P + initphase)
    u_ = np.sum(Cinit, 1)
    nu = len(u_)
    
    rnorm = np.zeros((1000,1))
    cnorm = np.zeros((1000,1))
    #U_init_abs = np.abs(np.fft.fft(u_))
    U_init_abs = np.abs(np.fft.rfft(u_))

    for ii in range(0,1000):
        threshold = np.log10(ii+137.8)/np.log10(1137.8)
        u_[u_>threshold] = threshold
        u_[u_<-threshold] = -threshold

        #U_ = np.fft.fft(u_)
        U_ = np.fft.rfft(u_)
        U_ = U_*U_init_abs/np.abs(U_)

        #u_ = np.fft.ifft(U_)
        u_ = np.fft.irfft(U_, n=nu)
        rnorm[ii] = np.linalg.norm(np.real(u_))
        cnorm[ii] = np.linalg.norm(np.imag(u_))
        #u_ = np.real(u_) # numerical stability

    #u_ = u_/np.max(np.abs(u_))
    return u_, t #, rnorm, cnorm


