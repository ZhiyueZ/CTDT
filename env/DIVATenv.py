import scipy.stats as stats
import scipy.integrate as integrate
from scipy.optimize import brentq
import numpy as np
from scipy.stats import expon


def sigmoid_k(theta_alpha,k):
  return(k/(1+np.exp(theta_alpha)))
  



class DIVATeval:
    def __init__(self, sigma2_l=0.1**2,beta_s1=1,
                        beta_s2=0.9,beta_s3=-0.75,beta_alpha=-5,h0=5,omega=1.05,mean_y_init=5,timeout=1000):

        self.time = 0
        self.steps_elapsed = 0
        self.tox = 0
        self.b_il = stats.multivariate_normal.rvs(mean=np.array([0,0,0]),
                                                  cov=np.array([0.2**2,0.07**2,1*10**(-8)]), size=1)
        self.DGF = stats.binom.rvs(n=1, p=0.4, size=1)[0]
        self.ageD = stats.norm.rvs(loc=0, scale=1, size=1)[0]
        self.BMI = stats.norm.rvs(loc=0, scale=1, size=1)[0]
        self.sigma2_l = sigma2_l
        self.y = stats.norm.rvs(loc=mean_y_init, scale=np.sqrt(self.sigma2_l), size=1)
        self.mean_y_init = mean_y_init
        # Ey, or y^*
        self.Ey = mean_y_init
        self.beta_s=beta_s1
        self.beta_sd=beta_s2
        self.beta_sd_cum=beta_s3
        self.beta_alpha = beta_alpha
        self.shape = omega
        self.h0=h0
        self.k=2
        self.theta_a=np.array([9.5,-1.5])
        self.beta_l=np.array([4,0.5,0.3,0.4, 0.25, -1*10**(-4),3*10**(-8)])
        self.timeout = timeout
        self.eta_tox=50
        self.alpha = sigmoid_k(np.dot(self.theta_a,np.array([1,self.y.item()])),self.k)
        self.censortime = 100000.
    def toxicity(self, t_upper, t_lower, di):
        ti = t_upper
        t_r = t_lower
        tox = self.tox*np.exp(-(ti-t_r)/self.eta_tox)
        weight=(1-np.exp(-(ti-t_r)/self.eta_tox))
        tox=tox+di*weight
        return tox
    def hazard_fun(self,t_upper,t_lower,di):
        ti = t_upper
        t_r = t_lower
        Ey_updated = self.Ey+(ti-t_r)*(self.beta_l[-2]+self.b_il[-1])+(ti-t_r)**2*self.beta_l[-1]
        tox_updated = ((self.tox*np.exp(-(ti-t_r)/self.eta_tox)) + (1-np.exp(-(ti-t_r)/self.eta_tox))*di)

        haz = (self.shape)*np.exp(-(self.beta_s*Ey_updated**2+self.beta_sd*di+ \
                self.beta_sd_cum*tox_updated+self.h0+self.beta_alpha*self.alpha))*ti**(self.shape-1)

        haz = np.nan_to_num(haz,posinf=0, neginf=-1e32)
        return haz

    def Ey_update(self,di,t_upper):
        ti=t_upper
        t_r = self.time
        Zvec = np.array([1,di,self.ageD, self.DGF, self.BMI, ti,ti**2])
        Rvec = np.array([1,di,ti])
        mean_fixed = np.dot(Zvec, self.beta_l)
        mean_rand = np.dot(Rvec, self.b_il)
        Ey_temp = mean_rand+mean_fixed
        return Ey_temp


    def cumulative_prob(self, t_upper, t_lower, di):
        I = integrate.quad(self.hazard_fun, t_lower, t_upper, args=(t_lower,di) )[0]
        cumu_prob = 1-np.exp(-I)
        return(cumu_prob)

    def inverse_cumu_prob(self, u, t_lower, di, delta_t):
        fun = lambda t: self.cumulative_prob(t,t_lower,di)-u
        #optimizer starts from t_lower because T_max>= t_lower
        T_max = brentq(fun,t_lower, (t_lower+delta_t))
        return(T_max)



    # soft reset; keep the same personal data
    def reset(self):
        self.y = stats.norm.rvs(loc=self.mean_y_init, scale=np.sqrt(self.sigma2_l), size=1)
        self.Ey = self.mean_y_init
        self.alpha = sigmoid_k(np.dot(self.theta_a,np.array([1,self.y.item()])),self.k)
        self.time = 0
        self.steps_elapsed=0
        self.tox = 0
        obs = self.get_obs()
        return obs
    
    # hard reset; redraw the random effect and personal data, or load from argument
    def hard_reset(self,info_i=None):
        if info_i:
            self.b_il = info_i['b_il']
            self.BMI = info_i['BMI']
            self.ageD = info_i['ageD']
            self.DGF = info_i['DGF']
        else:
            self.b_il = stats.multivariate_normal.rvs(mean=np.array([0,0,0]),
                                                  cov=np.array([0.2**2,0.07**2,1*10**(-8)]), size=1)
            self.DGF = stats.binom.rvs(n=1, p=0.4, size=1)[0]
            self.ageD = stats.norm.rvs(loc=0, scale=1, size=1)[0]
            self.BMI = stats.norm.rvs(loc=0, scale=1, size=1)[0]
        obs = self.reset()
        return obs


   

    def get_data_i(self):
        data_info = {}
        data_info['BMI'] = self.BMI
        data_info['ageD'] = self.ageD
        data_info['DGF'] = self.DGF
        data_info['b_il'] = self.b_il
        return data_info

    def get_obs(self):
        obs = np.concatenate([self.y,[self.ageD],[self.BMI],[self.DGF]])
        return obs



        # "thinning" algorithm
    # the death event is essentially the first event from an inhomogeneous poisson process
    def sample_event(self,di,delta_t):
        t_initial=self.time
        t_lower = self.time
        t_upper = self.time+delta_t
        #candidates from a homogeneous Poisson process
        t_candidate = t_lower
        
        # search for lambda upper bound in [t_lower,t_upper]
        t_range = np.linspace(t_lower+1e-8,t_upper,num=50)

        lambda_range=self.hazard_fun(t_range,t_lower,di) 
        
        
        lambda_bar = np.max(lambda_range)
        
        accepted = False
        t_accept = None
        if lambda_bar ==0:
            return accepted, t_accept
        # by here, lambda_bar >0 
        while t_lower<t_upper:
            #t_bar ~ exp(lambda_bar)
            t_bar = expon.rvs(scale=1/lambda_bar,size=1)[0]
            t_candidate =t_lower+t_bar
            
            lambda_candidate = self.hazard_fun(t_candidate,t_initial,di)
            u_s = stats.uniform.rvs(size=1)
            
            if u_s <= (lambda_candidate/lambda_bar) and t_candidate<t_upper:
                t_accept = t_candidate
                accepted = True
                break
            t_lower = t_candidate
        return accepted, t_accept


    def step(self, action):
        di = action[0]
        delta_t = action[1]

        Zvec = np.array([1,di,self.ageD, self.DGF, self.BMI, self.time,self.time**2])
        Rvec = np.array([1,di,self.time])
        mean_fixed = np.dot(Zvec, self.beta_l)
        mean_rand = np.dot(Rvec, self.b_il)
        self.Ey = mean_rand+mean_fixed


        self.steps_elapsed+=1

        t_ij = self.time+delta_t
        death = False
        self.alpha = sigmoid_k(np.dot(self.theta_a,np.array([1,self.y.item()])),self.k)
        death, T_max = self.sample_event(di,delta_t)
        #survive until t_ij
        if not death:
            if t_ij >= self.censortime:
                done = True
                reward = self.censortime - self.time
                Zvec = np.array([1,di,self.ageD, self.DGF, self.BMI, t_ij,t_ij**2])
                Rvec = np.array([1,di,t_ij])
                mean_fixed = np.dot(Zvec, self.beta_l)
                mean_rand = np.dot(Rvec, self.b_il)
                self.Ey = mean_rand+mean_fixed
                self.y = stats.norm.rvs(loc=self.Ey, scale=np.sqrt(self.sigma2_l), size=1)
                self.tox =  self.toxicity(t_ij, self.time, di)
                self.time = self.censortime
                
            else:
                done = False
                reward = delta_t
                Zvec = np.array([1,di,self.ageD, self.DGF, self.BMI, t_ij,t_ij**2])
                Rvec = np.array([1,di,t_ij])
                mean_fixed = np.dot(Zvec, self.beta_l)
                mean_rand = np.dot(Rvec, self.b_il)
                self.Ey = mean_rand+mean_fixed
                self.y = stats.norm.rvs(loc=self.Ey, scale=np.sqrt(self.sigma2_l), size=1)
                self.tox =  self.toxicity(t_ij, self.time, di)
                self.time = t_ij
        else:

            done = True
            Zvec = np.array([1,di,self.ageD, self.DGF, self.BMI, T_max,T_max**2])
            Rvec = np.array([1,di,T_max])
            mean_fixed = np.dot(Zvec, self.beta_l)
            mean_rand = np.dot(Rvec, self.b_il)
            self.Ey = mean_rand+mean_fixed
            self.y = stats.norm.rvs(loc=self.Ey, scale=np.sqrt(self.sigma2_l), size=1)
            reward = T_max - self.time
            self.time = T_max
     
        obs = self.get_obs()
        
        reward = np.array(reward)
        return obs, reward.item(), done, death



    def behavior(self,bias=0,noise=0.01):
        beta_d = np.array([-3, 1.2, 0.15, 0.2, 0.15])
        d_vec = np.array([1,self.y[0],self.ageD, self.DGF, self.BMI])
        d_mean = np.dot(d_vec,beta_d) - bias
        dosage = stats.norm.rvs(loc=d_mean, scale=np.sqrt(noise), size=1)[0]
        dosage = np.clip(dosage,1,None)
        dt_vec = np.array([dosage,self.time,self.y[0]])
        beta_dt = np.array([60,0.04,30])
        dt_mean = 800 - np.dot(dt_vec,beta_dt)
        
        dt = stats.norm.rvs(loc=dt_mean, scale=5, size=1)[0]
        dt = np.clip(dt,30,None)
        action_pair = np.array([dosage,dt])
        return action_pair
        