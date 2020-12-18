from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.integrate import solve_ivp
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from functools import partial
import multiprocessing as mp
import pandas as pd
from sklearn.linear_model import MultiTaskLassoCV , Lasso

    
def mae(pred, true):
    return np.mean(np.abs(pred - true))


class SIR_fitter():
    ''' Class that use the features to extract SIR parameters. Fitted parameters will be keyed by country and date. 
    infected_days: days previous to day0 to sum the number of currently infected.
    semi_fit_days: number of days before and after the actual day to fit the SIR parameters on.
    beta_i: initial value for fitting beta.
    gamma_i: initial value for fitting gamma.
    '''
    def __init__(self, moving_average=False, 
                 infection_days=7, semi_fit_days=7,
                 beta_i=0.6, gamma_i=1/7,nprocs=4):
        
        self.infection_days=infection_days
        self.semi_fit=semi_fit_days
        self.fit_days=semi_fit_days*2+1
        self.time_integ=np.linspace(-self.semi_fit,self.semi_fit,self.fit_days)
        self.beta_i=beta_i
        self.gamma_i=gamma_i
        self.nprocs=nprocs
        self.moving_average=moving_average
        
    def fit_country(self,df_country):
        COL = ['NewCases'] if not self.moving_average else ['MA']
        X_cols = df_country.columns
        y_col = COL
        gdf = df_country
        gdf['beta']=np.nan
        gdf['gamma']=np.nan
        all_case_data = np.array(gdf[COL])
        nb_total_days = len(gdf)
        for d,(idx,row) in enumerate(gdf.iterrows()):
            if d>self.semi_fit and d<(nb_total_days-self.semi_fit):
                N=row.Population
                X_cases = all_case_data[d - self.semi_fit:d+self.semi_fit+1].reshape(-1)
                I0=all_case_data[d-self.semi_fit- self.infection_days: 
                                                       d-self.semi_fit+1].sum()
                R0=all_case_data[0:(d-self.semi_fit- self.infection_days)].sum()
                Ic0=all_case_data[0:(d-self.semi_fit)].sum()
                S0=N-I0-R0
                x0=(S0,I0,Ic0,R0)
                if I0<0:
                    raise ValueError('Infected was {} for popolation {}'.format(x0[1],X_i[self.lookback_days+1]))
                elif I0<1:
                    popt=np.array([np.nan,np.nan])
                else:
                    fintegranda=partial(self.__SIR_integrate,self.time_integ,x0,N)
                    popt, pcov = curve_fit(fintegranda, self.time_integ, 
                           X_cases[1:],
                           p0=[self.beta_i,self.gamma_i],maxfev=5000,bounds=([0.,0.],
                                                           [np.inf,1.]))
                    gdf.loc[idx,'beta']=popt[0]
                    gdf.loc[idx,'gamma']=popt[1]                
        return gdf
        
        
    def __SIR_ode(self,t,x0, N, beta, gamma):
        S, I, Ic, R = x0
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        # Computes the cumulative new cases during the integration period
        dIc= -dS
        dR = gamma * I
        return dS, dI, dIc,dR
    
    def __SIR_integrate(self,ttotp,x0,N,ti,beta,gamma):
        ''' Argument ti not used but needed by curve_fit '''
        sol=solve_ivp(self.__SIR_ode,[ttotp[0],ttotp[-1]],x0,args=(N,beta,gamma),t_eval=ttotp)
        #lung=len(sol.y[0])
        # The only variable to predict is "NewCases", i.d. the difference of the cumulative Ic
        return np.diff(sol.y[2])#.reshape(lung,1).flatten()
    
   
        
   
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
    
    def fit(self,df,save_to=None):
        '''
        Fit SIR parameters on all the data. 
        save_to: path to save the results in pickle format. Results are saved a Pandas DataFrame having columns: GeoID,Date,beta,gamma 
        '''
        if self.semi_fit<3:
            raise ValueError('ValueError: semi_fit_days should be higher than 2')
        geo_ids = df.GeoID.unique()
        COL = ['NewCases'] if not self.moving_average else ['MA']
        df=df[['GeoID','Date','Population']+COL]
        self.df_chunks=[df[df.GeoID==g].sort_values('Date') 
                        for g in geo_ids]
        nchunks=len(self.df_chunks)
        pool=mp.Pool(self.nprocs)
        outputs=list(tqdm(pool.imap(self.fit_country,self.df_chunks),total=nchunks))
        pool.close()
        pool.join()
        self.df_pars=pd.concat(outputs)
        self.df_pars.sort_values(['GeoID','Date'],inplace=True)
        if save_to is not None:
            with open(save_to,'wb') as f:
                pickle.dump(self.df_pars,f)
                
        # Return the classifier
        return self

class SIR_predictor(BaseEstimator, RegressorMixin, SIR_fitter):
    def __init__(self, df,moving_average=False, 
                 infection_days=7, semi_fit_days=7,
                 beta_i=0.6, gamma_i=1/7,lookback_days=15,
                 ML_model='LassoCV(max_iter=100000,tol=1e-7)',nprocs=4):
        self.df=df
        self.moving_average=moving_average
        self.infection_days=infection_days
        self.semi_fit=semi_fit_days
        self.beta_i=beta_i
        self.gamma_i=gamma_i
        self.lookback_days=lookback_days
        self.nprocs=nprocs
        self.MLmodel=eval(ML_model)
    
    def SIR_ode(self,t,x0, N, beta, gamma):
        return self._SIR_fitter__SIR_ode(t,x0, N, beta, gamma)
    def SIR_integrate(self,ttotp,x0,N,ti,beta,gamma):
        return self._SIR_fitter__SIR_integrate(ttotp,x0,N,ti,beta,gamma)
    def fit(self,X,y):
        check_X_y(X,y)
        
        self.SFmodel=SIR_fitter(self.moving_average, 
                 self.infection_days, self.semi_fit,
                 self.beta_i, self.gamma_i,self.nprocs)
        print('Fitting SIR parameters...')
        self.SFmodel.fit(self.df)
        #self.SFmodel.df_pars=self.df.copy()
        #self.SFmodel.df_pars['beta']=0.6
        #self.SFmodel.df_pars['gamma']=1/7
        self.df=self.df.merge(self.SFmodel.df_pars[['GeoID','Date','beta','gamma']],
                 how='left',on=['GeoID','Date'],left_index=True).dropna(subset=['beta','gamma'])
        #print(self.df.loc[X[:,-1]])
        
        #Predict SIR parameters instead of cases
        #Use only row in training set AND with SIR pars not being nans
        idx=np.array(list(set(X[:,-1]).intersection(set(self.df.index))),
                     dtype=int).reshape(-1)
        #print(len(idx),idx[:10])
        self.y_pars=np.array(self.df.loc[idx, ['beta','gamma']])
        #print(self.y_pars.shape)
        self.X_pars=X[np.in1d(X[:,-1],idx)]
        #print(self.X_pars.shape)
        # remove last column (df.index)
        # remove first lookback_days columns: not using cases to predict parameters
        self.X_pars=self.X_pars[:,self.lookback_days:-1]
        
        self.MLmodel.fit(self.X_pars,self.y_pars)
        self.TMAE=mae(self.MLmodel.predict(self.X_pars),self.y_pars)
        #print('Training MAE for params:', self.TMAE)
        return self
    
    def predict_pars(self,X):
        return self.MLmodel.predict(X[:,self.lookback_days:-1])
    
    def predict_chunk(self,X_chunk):
        y_chunk=[]
        for i in range(X_chunk.shape[0]):
            N=X_chunk[i,self.lookback_days+1]
            I0=X_chunk[i,self.lookback_days-self.infection_days:self.lookback_days].sum()
            Ic0=X_chunk[i,self.lookback_days]
            R0=Ic0-I0
            S0=N-I0-R0
            x0=(S0,I0,Ic0,R0)
            pars=self.predict_pars(X_chunk[i,:].reshape(1,-1))
            beta=pars[0][0]
            gamma=pars[0][1]
            time_integ=[0,1]
            I=self.SIR_integrate(time_integ,x0,N,time_integ,beta,gamma)[0]
            y_chunk.append(I)
        return y_chunk
    
    def predict(self,X):
        if X.shape[0]>1:
            nchunks=self.nprocs*10
            X_chunks=np.array_split(X,nchunks)
            pool=mp.Pool(self.nprocs)
            y_chunks=pool.map(self.predict_chunk,X_chunks)
            pool.close()
            pool.join()
            y=[item for sublist in y_chunks for item in sublist]
            return y
        else:
            return np.nan

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
    ''' DO NOT USE: Class that use the features to extract SIR parameters. Fitted parameters will be keyed by country and date. 
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
        ''' Argument ti not used but needed by curve_fit '''
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
    
