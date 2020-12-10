from sklearn.base import BaseEstimator, RegressorMixin
class SIRRegressor(BaseEstimator, RegressorMixin):
    ''' Model that use the features to extract SIR parameters and compute predictions. 
    single_pred_days: number of next days to predict. Predictions could be of more than one day, so to have a multi-variate regression (NOT TESTED YET)'''
    def __init__(self, single_pred_days=1):
        from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
        from scipy.integrate import solve_ivp
        #self.demo_param = demo_param
        self.params=None
        self.single_pred_days=single_pred_days
        
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
            # Division is to ensure values lower than 1, but should be removed when there will be a real training
            beta=self.params[0].dot(X_normed[i])/X.shape[1]
            gamma=self.params[1].dot(X_normed[i])/X.shape[1]
            N=X[i,lookback_days+1]
            I0=X[i,lookback_days-infection_days:lookback_days-1].sum()
            R0=X[i,lookback_days]
            S0=N-I0-R0
            x0=(S0,I0,R0,R0)
            Ipred=self.__SIR_integrate(time_integ,x0,N,beta,gamma)[-1]
            y_pred.append(Ipred)
        
        return np.array(y_pred)