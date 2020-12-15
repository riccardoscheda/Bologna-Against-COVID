from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from functools import partial
import multiprocessing as mp


class SIRRegressor(BaseEstimator, RegressorMixin):
    ''' Model that use the features to extract SIR parameters and compute predictions. 
    single_pred_days: number of next days to predict. Predictions could be of more than one day, so to have a multi-variate regression (NOT TESTED YET)'''
    def __init__(self, single_pred_days=1,lookback_days=30,infection_days=15):
        
        #self.demo_param = demo_param
        self.params=None
        self.single_pred_days=single_pred_days
        self.lookback_days=lookback_days
        self.infection_days=infection_days
        
    def __SIR_ode(self,t,x0, N, beta, gamma):
        S, I, Ic, R = x0
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        # Computes the cumulative new cases during the integration period
        dIc= -dS
        dR = gamma * I
        return dS, dI, dIc,dR
    
    def __SIR_integrate(self,ttotp,x0,N,beta,gamma):
        sol=solve_ivp(self.__SIR_ode,[ttotp[0],ttotp[-1]],x0,args=(N,beta,gamma),t_eval=ttotp)
        #lung=len(sol.y[0])
        # The only variable to predict is "NewCases", i.d. the difference of the cumulative Ic
        return np.diff(sol.y[2],prepend=x0[2])#.reshape(lung,1).flatten()
        
        
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        
        #Assign random parameters, to be used for scalar product with the features.
        # We sample two rows: one to get beta and one to get gamma
        self.params=np.random.uniform(0,1,(2,X.shape[1]))
        # Return the classifier
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        
        # Normalization is to have small values also with random parameters (to be removed later maybe)
        X_normed = (X - X.min(0)) / X.ptp(0)
        X_normed=np.nan_to_num(X_normed)
        
        y_pred=[]
        time_integ=np.linspace(0,self.single_pred_days,self.single_pred_days+1)
        for i in tqdm(range(X.shape[0]),total=X.shape[0]):
            # Apply ML model parameters to get SIR parameters
            # Division for X.shape[1]  should ensure values lower than 1, but should be removed when there will be a real training
            beta=self.params[0].dot(X_normed[i])/X.shape[1]
            gamma=self.params[1].dot(X_normed[i])/X.shape[1]
            
            # Total population
            N=X[i,self.lookback_days+1]
            
            # Currently infected individuals
            I0=X[i,self.lookback_days-self.infection_days:self.lookback_days-1].sum()
            
            # Recovered individuals (taken as current total confirmed cases)
            R0=X[i,self.lookback_days]
            
            # Susceptible individuals
            S0=N-I0-R0
            
            # Initial condition of integration
            x0=(S0,I0,R0,R0)
            Ipred=self.__SIR_integrate(time_integ,x0,N,beta,gamma)[-1]
            y_pred.append(Ipred)
        
        return np.array(y_pred)
    


class SIR_parfinder(BaseEstimator, RegressorMixin):
    ''' Model that use the features to extract SIR parameters and compute predictions. 
    single_pred_days: number of next days to predict. Predictions could be of more than one day, so to have a multi-variate regression (NOT TESTED YET)
    lookback_days: past days to use. Must be higher than fit_days+infected_days.
    infected_days: days previous to day0 to sum the number of currently infected.
    fit_days: number of days before the last day to fit the SIR parameters on (only for labelization).
    beta_i: initial value for fitting beta (only for labelization).
    gamma_i: initial value for fitting gamma (only for labelization).
    '''
    def __init__(self, single_pred_days=1,lookback_days=30,infection_days=7,fit_days=15,beta_i=0.6,gamma_i=1/14,nprocs=4):
        
        #self.demo_param = demo_param
        self.single_pred_days=single_pred_days
        self.lookback_days=lookback_days
        self.infection_days=infection_days
        self.fit_days=fit_days
        self.time_integ=np.linspace(lookback_days-fit_days,lookback_days,fit_days+1)
        self.beta_i=beta_i
        self.gamma_i=gamma_i
        self.nprocs=nprocs
        
        
    def __SIR_ode(self,t,x0, N, beta, gamma):
        S, I, Ic, R = x0
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        # Computes the cumulative new cases during the integration period
        dIc= -dS
        dR = gamma * I
        return dS, dI, dIc,dR
    
    def __SIR_integrate(self,ttotp,x0,N,ti,beta,gamma):
        ''' Argument ti not used but needed by curve_fit'''
        sol=solve_ivp(self.__SIR_ode,[ttotp[0],ttotp[-1]],x0,args=(N,beta,gamma),t_eval=ttotp)
        #lung=len(sol.y[0])
        # The only variable to predict is "NewCases", i.d. the difference of the cumulative Ic
        return np.diff(sol.y[2])#.reshape(lung,1).flatten()
    
    def labelize_chunk(self,i):
        X_chunk=self.X_chunks[i]
        labels=np.empty((X_chunk.shape[0],2))
        for i in range(X_chunk.shape[0]):
            labels[i,:]=self.row_fit(X_chunk[i,:])
        return (i,labels)        
        
    def row_initial_conditions(self,X_i):
        '''
        Returs the initial condition from the the SIR integration of this row.
        '''
        N=X_i[self.lookback_days+1]            
        # Currently infected individuals (sum of new cases on the previous infection_days before the first fit day)
        I0=X_i[self.lookback_days-self.fit_days-self.infection_days:self.lookback_days-self.fit_days+1].sum()            
        # Recovered individuals (taken as current total confirmed cases)
        R0=X_i[self.lookback_days]        
        # Susceptible individuals
        S0=N-I0-R0            
        # Initial condition of integration
        x0=(S0,I0,R0,R0)
        return N,x0
    
    def row_observed_cases(self,X_i):
        '''
        Return an array with length fit_days with the real observed new_cases of that row.
        '''
        return X_i[self.lookback_days-self.fit_days:self.lookback_days]
    
    def row_fit(self,X_i):
        '''
        Fit SIR parameters for one observation
        '''
        N,x0=self.row_initial_conditions(X_i)
        if x0[1]<0:
            raise ValueError('Infected was {} for popolation {}'.format(x0[1],X_i[self.lookback_days+1]))
        elif x0[1]<1:
            popt=np.array([np.nan,np.nan])
        else:
            fintegranda=partial(self.__SIR_integrate,self.time_integ,x0,N)
            popt, pcov = curve_fit(fintegranda, self.time_integ, 
                       self.row_observed_cases(X_i),
                       p0=[self.beta_i,self.gamma_i],maxfev=5000,bounds=([0.,0.],
                                                           [np.inf,1.]))
        return popt
        
    def row_predict(self,X_i,pars):
        '''
        Given the SIR parameters, predicts the new cases. The last one is the actual prediction for the final MAE
        '''
        N,x0=self.row_initial_conditions(X_i)
        beta=pars[0]
        gamma=pars[1]
        Ipred=self.__SIR_integrate(self.time_integ,x0,N,self.time_integ,beta,gamma)
        return Ipred
    
    def fit(self, X):
        if self.fit_days+self.infection_days>self.lookback_days:
            raise ValueError('ValueError: lookback_days ({}) must be higher than fit_days+infected_days ({})'.format(
                self.lookback_days,self.infection_days+self.fit_days))
        if self.fit_days<5:
            raise ValueError('ValueError: fit_days should be higher than 4')
        X = check_array(X)
        
        nchunks=self.nprocs*10
        self.X_chunks = [X[n * len(X) // nchunks : (n + 1) * len(X) // nchunks] for n in range(nchunks)]
        #nchunks=len(self.X_chunks)
        pool=mp.Pool(self.nprocs)
        outputs=list(tqdm(pool.imap(self.labelize_chunk,range(nchunks)),total=nchunks))
        pool.close()
        pool.join()
        outputs.sort(key=lambda x:x[0])
        self.labels_=np.concatenate([chunk_lab[1] for chunk_lab in outputs])
        # Return the classifier
        return self
    
    def predict(self, X):
        '''
        Fit SIR parameters for each observation
        '''
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        if X.shape[1]==1:
            return self.row_fit(X.reshape(-1))
        if X.shape[0]>self.nprocs*10:
            nchunks=self.nprocs*10
        else:
            nchunks=1
        self.X_chunks = [X[n * len(X) // nchunks : (n + 1) * len(X) // nchunks] for n in range(nchunks)]
        #nchunks=len(self.X_chunks)
        pool=mp.Pool(self.nprocs)
        outputs=list(tqdm(pool.imap(self.labelize_chunk,range(nchunks)),total=nchunks))
        pool.close()
        pool.join()
        outputs.sort(key=lambda x:x[0])
        return np.concatenate([chunk_lab[1] for chunk_lab in outputs])