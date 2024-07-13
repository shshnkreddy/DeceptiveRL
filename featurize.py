import numpy as np

def featurize(s, s_size, a = 0, a_size = 0, n = 50, basis = 'gaussian'):
    f_s = None
    if(basis == 'poly'):
        f = np.ones(n+1) * (s+1)
        p = np.arange(0, n+1)
        f_s = np.power(f,p)

    elif(basis == 'gaussian'):
        mu = np.linspace(0, s_size, n)
        sigma = np.ones(n)*0.5

        f_s = np.exp(-(s - mu) ** 2 / (2 * (sigma ** 2))) / np.sqrt(2 * np.pi * (sigma ** 2))

    elif(basis == 'simple'):
        f_s = np.array([1, s])

    elif(basis == 'rbf'):
        l = 2
        mus = np.linspace(0, s_size, n)
        mua = np.arange(0, a_size)

    #shape (a,s)
        mu_s, mu_a = np.meshgrid(mus, mua)
        mu_s = mu_s.flatten()
        mu_a = mu_a.flatten()

        f_s = np.zeros_like(mu_s)
        cnt = 0
        for i in range(mu_s.shape[0]):
            f_s[cnt] = np.exp(-(np.linalg.norm(np.array([s,a]) - np.array([mu_s[i], mu_a[i]]))) / l) ** 2
            cnt += 1

    elif(basis == 'dgaussian'):
        mu = np.linspace(0, s_size, n)
        sigma = np.ones(n) * 0.3

        f_s = np.exp(-(s - mu) ** 2 / (2 * (sigma ** 2))) / np.sqrt(2 * np.pi * (sigma ** 2))

        f_s = np.concatenate((f_s, -f_s))
    
    elif(basis == 'drbf'):
        l = 2
        mus = np.linspace(0, s_size, n)
        mua = np.arange(0, a_size)

        #shape (a,s)
        mu_s, mu_a = np.meshgrid(mus, mua)
        mu_s = mu_s.flatten()
        mu_a = mu_a.flatten()

        f_s = np.zeros(n)
        cnt = 0
        for i in range(n):
            f_s[cnt] = np.exp(-(np.linalg.norm(np.array([s,a]) - np.array([mu_s[i], mu_a[i]]))) / l) ** 2
            f_s[cnt+1] = -np.exp(-(np.linalg.norm(np.array([s,a]) - np.array([mu_s[i], mu_a[i]]))) / l) ** 2
            cnt += 2

    elif(basis == 'fourier'):
        s /= s_size
        f_s = np.cos(np.arange(0,n)*s*np.pi)

    return f_s