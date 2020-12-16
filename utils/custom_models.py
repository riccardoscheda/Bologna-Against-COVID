from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from functools import partial
import multiprocessing as mp
import pandas as pd


#Questa non ha il fit, non è usabile
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
    


class SIR_parfinder():
    ''' Class that use the features to extract SIR parameters. Fitted parameters will be keyed by country and date. 
    single_pred_days: number of next days to predict. Predictions could be of more than one day, so to have a multi-variate regression (NOT TESTED YET)
    lookback_days: past days to use. Must be higher than fit_days+infected_days.
    infected_days: days previous to day0 to sum the number of currently infected.
    fit_days: number of days before the last day to fit the SIR parameters on (only for labelization).
    beta_i: initial value for fitting beta (only for labelization).
    gamma_i: initial value for fitting gamma (only for labelization).
    '''
    def __init__(self, df,moving_average=False, 
                 infection_days=7, semi_fit_days=7,
                 beta_i=0.6, gamma_i=1/7,nprocs=4):
        
        self.infection_days=infection_days
        self.semi_fit=semi_fit_days
        self.fit_days=semi_fit_days*2+1
        self.time_integ=np.linspace(-self.semi_fit,self.semi_fit,self.fit_days)
        self.beta_i=beta_i
        self.gamma_i=gamma_i
        self.nprocs=nprocs
        self.df=self.__add_lookback(df,moving_average)
    
    def __add_lookback(self,df,moving_average):
        COL = ['NewCases'] if not moving_average else ['MA']
        X_cols = df.columns
        y_col = COL
        geo_ids = df.GeoID.unique()
        fit_ids=['CasesDay{}'.format(d) for d in range(-self.semi_fit,self.semi_fit+1)]
        self.fit_ids=fit_ids
        #print(fit_ids)
        df=df[['GeoID','Date','Population']+COL]
        df=pd.concat([df,pd.DataFrame(columns=fit_ids+['I0','R0'])])
        print('Adding lookback days to the dataframe...')
        for g in tqdm(geo_ids):
            gdf = df[df.GeoID == g]
            all_case_data = np.array(gdf[COL])
            nb_total_days = len(gdf)
            for d,(idx,row) in enumerate(gdf.iterrows()):
                if d>self.semi_fit and d<(nb_total_days-self.semi_fit):
                    X_cases = all_case_data[d - self.semi_fit:d+self.semi_fit+1].reshape(-1)
                    try:
                        df.loc[idx,fit_ids]=X_cases
                        df.loc[idx,'I0']=all_case_data[d-self.semi_fit- self.infection_days: 
                                                       d-self.semi_fit+1].sum()
                        df.loc[idx,'R0']=all_case_data[0: 
                                                       (d-self.semi_fit-
                                                       self.infection_days)].sum()
                    except ValueError:
                        print(row.GeoID)
                        print(row.Date)
                        print(df.loc[idx-self.semi_fit,fit_ids].shape)
                        print(X_cases.shape)
                        raise ValueError('Mismatch in shapes for this entry, check the code...')
                        
        return df.dropna()
        
        
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
    
    def labelize_chunk(self,df_chunk):
        #df_chunk=self.df_chunks[i]
        pars_df=df_chunk[['GeoID','Date']].copy()
        pars_df['beta']=np.nan
        pars_df['gamma']=np.nan
        for j,(idx,row) in enumerate(df_chunk.iterrows()):
            pars_df.iloc[j,:]=[row.GeoID,row.Date]+list(self.row_fit(row))
        return pars_df        
        
    def row_initial_conditions(self,row):
        '''
        Returns the initial condition from the the SIR integration of this row.
        '''
        N=row.Population           
        # Currently infected individuals (sum of new cases on the previous infection_days before the first fit day)
        I0=row.I0          
        # Recovered individuals (taken as current total confirmed cases)
        R0=row.R0     
        # Susceptible individuals
        S0=N-I0-R0            
        # Initial condition of integration
        x0=(S0,I0,R0,R0)
        return N,x0
    
    def row_observed_cases(self,row):
        '''
        Return an array with length fit_days with the real observed new_cases of that row.
        '''
        return row[self.fit_ids[1:]].values
    
    def row_fit(self,row):
        '''
        Fit SIR parameters for one observation
        '''
        N,x0=self.row_initial_conditions(row)
        if x0[1]<0:
            raise ValueError('Infected was {} for popolation {}'.format(x0[1],X_i[self.lookback_days+1]))
        elif x0[1]<1:
            popt=np.array([np.nan,np.nan])
        else:
            fintegranda=partial(self.__SIR_integrate,self.time_integ,x0,N)
            popt, pcov = curve_fit(fintegranda, self.time_integ, 
                       self.row_observed_cases(row),
                       p0=[self.beta_i,self.gamma_i],maxfev=5000,bounds=([0.,0.],
                                                           [np.inf,1.]))
        return popt.reshape(-1)
        
    def row_predict(self,GeoID,date,pars=None):
        '''
        Given the SIR parameters, predicts the new cases. The last one is the actual prediction for the final MAE
        '''
        row=self.df.loc[(self.df.GeoID==GeoID)&(self.df.Date==pd.to_datetime(date)),:].iloc[0,:]
        N,x0=self.row_initial_conditions(row)
        if np.isnan(pars[0]):
            return np.repeat(row.CasesDay0, self.fit_days)[1:]
        if pars is not None:
            beta=pars[0]
            gamma=pars[1]
        else:
            beta=self.df_pars.loc[(self.df_pars.GeoID==GeoID) & 
                                  (self.df_pars.Date==pd.to_datetime(date)),'beta'].iloc[0]
            gamma=self.df_pars.loc[(self.df_pars.GeoID==GeoID) & 
                                  (self.df_pars.Date==pd.to_datetime(date)),'gamma'].iloc[0]
        Ipred=self.__SIR_integrate(self.time_integ,x0,N,self.time_integ,beta,gamma)
        return Ipred
    
    def fit(self,save_to=None):
        '''
        Fit SIR parameters on all the data. 
        save_to: path to save the results in pickle format. Results are saved a Pandas DataFrame having columns: GeoID,Date,beta,gamma 
        '''
        if self.semi_fit<3:
            raise ValueError('ValueError: semi_fit_days should be higher than 2')
        
        nchunks=self.nprocs*10
        self.df_chunks = np.array_split(self.df,nchunks)
        #print(type(self.df_chunks))
        empty_cunks=[df for df in self.df_chunks if df.shape[0]==0]
        if len(empty_cunks):
            print('{} empty chunks'.format(len(empty_chunks)))
        #nchunks=len(self.X_chunks)
        pool=mp.Pool(self.nprocs)
        outputs=list(tqdm(pool.imap(self.labelize_chunk,self.df_chunks),total=nchunks))
        pool.close()
        pool.join()
        self.df_pars=pd.concat(outputs)
        self.df_pars.sort_values(['GeoID','Date'],inplace=True)
        if save_to is not None:
            with open(save_to,'wb') as f:
                pickle.dump(self.df_pars,f)
                
        # Return the classifier
        return self
    
    