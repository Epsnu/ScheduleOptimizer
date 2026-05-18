import pandas as pd
import numpy as np

class Schedule:
    def __init__(self, pref_file, shift_file):
        # Load CSV files
        df_prefs = pd.read_csv(pref_file)
        df_shifts = pd.read_csv(shift_file)
        
        # Read member names
        self.member_names = df_prefs['name'].to_numpy()
        # Read shift names
        shift_names = []
        for i in range(1,df_prefs.shape[1]):
            shift_names.append(df_prefs.columns[i])
        self.shift_names = np.array(shift_names)

        # Constants n and k
        n = df_prefs.shape[0]
        k = df_shifts.shape[0]

        # Read worker preference matrix P
        P = df_prefs[:,df_prefs.columns.get_loc('name')+1:].to_numpy()

        # Determine and Read shift matrix S
        S = np.array((len(shift_names),2))
        for i in range(len(shift_names)):
            S[i,:] = df_shifts['hours','num_workers'].where(df_shifts['name']==shift_names[i])

        # max hours per worker x
        x = max(
            np.ceil(1/n*sum(S[:,0]*S[:,1])),
            max(S[:,0])
        )

        self.n = n
        self.k = k
        self.P = P
        self.S = S
        self.x = x
    
    # objective function
    def evaluate(self, theta, lambda_arr, sigma_arr):
        Q = self.f(theta)
        Q += lambda_arr[0]*self.Penalty_C1(theta, sigma_arr[0])
        Q += lambda_arr[1]*self.Penalty_C2(theta, sigma_arr[1])
        Q += lambda_arr[2]*self.Penalty_C3(theta)
        Q += lambda_arr[3]*self.Penalty_C4(theta, sigma_arr[2])
        return Q

    # helper P_hat function
    def P_hat_i(self,i,s):
        alpha = s - np.floor(s)
        P_hat_i = (1 - alpha)*self.P[i,np.floor(s)] + alpha*self.P[i,np.ceil(s)]
        return P_hat_i

    # cost function
    def f(self, theta):
        f = 0
        for i in range(self.n):
            for j in range(self.x):
                f -= self.P_hat_i(i,theta[i,j])
        return f
    
    # kernel density estimator
    def K(sigma, phi):
        return np.exp(-(phi)^2/(2*sigma*sigma))
    
    # penalty helper
    def C_im(self, theta, m, i, sigma):
        C_im = 0
        for j in range(self.x):
            C_im += self.K(sigma, theta[i,j] - m)
        return C_im
    
    # penalty C1
    def Penalty_C1(self, theta, sigma):
        Penalty_C1 = 0
        for m in range(self.k):
            for i in range(self.n):
                Penalty_C1 += (self.C_im(theta, m, i, sigma) - (self.S[m,0]*self.S[m,1]))^2
        return Penalty_C1
    
    # penalty C2
    def Penalty_C2(self, theta, sigma):
        Penalty_C2 = 0
        for m in range(self.k):
            for i in range(self.n):
                Penalty_C2 += max(0, self.C_im(theta, m, i, sigma) - self.S[m,0])^2
        return Penalty_C2
    
    # penalty C3
    def Penalty_C3(self, theta, tau=1):
        Penalty_C3 = 0
        for i in range(self.n):
            for j in range(self.x):
                Penalty_C3 += max(0, tau - self.P_hat_i(i,theta[i,j]))^2
        return Penalty_C3
    
    # penalty C4
    def Penalty_C4(self, theta, sigma):
        Penalty_C4 = 0
        for m in range(self.k):
            for i in range(self.n):
                Penalty_C4 += self.C_im(theta, m, i, sigma)^2*(self.C_im(theta, m, i, sigma) - self.S[m,0])^2
        return Penalty_C4
        

    # Normalize rows by mean of row
    def normalizePrefs(self):
        self.P = self.P / self.P.mean(axis=1, keepdims=True)