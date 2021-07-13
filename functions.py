### Refer to the Jupyter notebook for the maths behind these functions ###

def trapezoidal(a, b, function, n=100):
    
    """
    Parameters
    ----------
    a: Left end-point of the interval.
    b: Right end-point of the interval.
    function: The function for which we approximate its integral over [a,b].
    n: Number of points to be used for the partitioning of [a,b].
    
    Returns
    -------
    I: Integral approximation using the Trapezoidal rule.
    
    """
    x = np.linspace(a, b, n + 1) # n+1 points to create n subintervals
    y = function(x)
    y_right = y[1:] 
    y_left = y[: -1] 
    h = (b - a) / n
    I = (h / 2) * np.sum(y_right + y_left) # Using the summation formula marked with (*)
    
    return I


def BS_analytical_price(K, r, T, σ, S0, a, b):
    
    """
    Parameters
    ----------
    K: Fixed payoff of the option when it is in-the-money.
    r: The non-negative constant interest rate.
    T: Maturity date.
    σ: Volatility of the stock (must be positive).
    S0: Initial stock price (must be positive).
    a: Lower end-point of the in-the-money interval of the option.
    b: Upper end-point of the in-the-money interval of the option.
    
    Returns
    -------
    Time-0 price of the option calculated by the analytical formula in (a).
    
    """ 
    lower = ( np.log(a / S0) - (r - 0.5 * (σ**2)) * T  ) / (σ * T**0.5) # lower end-point to use in Normal CDF
    upper = ( np.log(b / S0) - (r - 0.5 * (σ**2)) * T  ) / (σ * T**0.5) # upper end-point to use in Normal CDF
    phi_lower = sc.stats.norm.cdf(x = lower , loc = 0, scale = 1) # Standard Normal CDF applied at lower end-point
    phi_upper = sc.stats.norm.cdf(x = upper , loc = 0, scale = 1) # Standard Normal CDF applied at upper end-point
    
    
    return (K * np.exp(-r * T)) * (phi_upper - phi_lower) # time-0 price of the option using part (a) analytical formula



def standard_MC_option_pricing(nsamples, seed, c_level, K, r, T, σ, S0, a, b):
    
    """
    Parameters
    ----------
    nsamples: Number of simulated random variables to be used in the Monte Carlo calculations.
    seed: Seed for random number generation reproducibility.
    c_level: Level of the confidence interval (i.e. use 0.95 for a 95% confidence interval), should be in (0,1).
    K: Fixed payoff of the option when it is in-the-money (must be positive).
    r: The non-negative constant interest rate.
    T: Maturity date (must be positive).
    σ: Volatility of the stock (must be positive).
    S0: Initial stock price (must be positive).
    a: Lower end-point of the in-the-money interval of the option (must be positive).
    b: Upper end-point of the in-the-money interval of the option (must be greater than a).
    
    Returns
    -------
    MC_option_price : Monte Carlo estimate for the time-0 option price, as described in (b.1).
    MC_option_price_var: Variance of the standard Monte Carlo estimate for the time-0 option price.
    confidence_interval: Asymptotic ('c_level' * 100)% confidence interval for the time-0 option price.
    
    """
    
    rng = np.random.default_rng(seed = seed)
    alpha = sc.stats.norm.ppf(q = 1 - ( (1 - c_level) / 2) , loc = 0, scale = 1) # Used in the confidence interval
    
    mu_log_S = np.log(S0) + (r - (0.5 * σ**2)) * T # mean of log(S_T)
    sigma_log_S = ( (σ**2) * T) ** 0.5 # standard deviation of log(S_T)
    log_stock_sample = rng.normal(loc = mu_log_S, scale = sigma_log_S, size = nsamples) # log(S_T) random sample
    
    # MC estimator for Q[ log(a) < log(S_T) < log(b) ]
    I_MC = sum( np.logical_and( np.log(a) < log_stock_sample , log_stock_sample < np.log(b)) ) / nsamples
    
    I_MC_std = ( (I_MC * (1 - I_MC)) / nsamples ) ** 0.5 # standard deviation of I_MC
    
    MC_option_price = (K * np.exp(-r * T)) * I_MC # MC estimate for the option price
     
    MC_option_price_std = (K * np.exp(-r * T)) * I_MC_std # standard deviation of option price MC estimate
    
    MC_option_price_var = MC_option_price_std**2 # variance of option price MC estimate
    
    confidence_interval = [  ( MC_option_price - (alpha * MC_option_price_std) ) , 
                            ( MC_option_price + (alpha * MC_option_price_std) ) ]
    
    
    return MC_option_price , MC_option_price_var, confidence_interval




def antithetic_option_pricing(nsamples, seed, K, r, T, σ, S0, a, b, corr_display = False):
    
    """
    Parameters
    ----------
    nsamples: Number of simulated random variables to be used in the Monte Carlo calculations
    seed: Seed for random number generation reproducibility.
    K: Fixed payoff of the option when it is in-the-money
    r: The non-negative constant interest rate
    T: Maturity date
    σ: Volatility of the stock (must be positive)
    S0: Initial stock price
    a: Lower end-point of the in-the-money interval of the option
    b: Upper end-point of the in-the-money interval of the option
    corr_display: Set this to True if you want the function to print the sample correlation between the 
                  1{a < S_i(+) < b} and 1{a < S_i(-) < b} random variables.
    
    Returns
    -------
    antithetic_estimate_price : Antithetic variates estimate for the time-0 option price
    antithetic_estimate_var : Variance of the Antithetic variates estimate for the time-0 option price
    
    """
    #Use half the nsamples for the random sample for fair comparison with the standard MC estimator
    nsamples = nsamples // 2 
    
    rng = np.random.default_rng(seed = seed)
    z_i = rng.standard_normal(size = nsamples)
    
    s_plus = S0 * np.exp((r - σ**2/2) * T + σ * np.sqrt(T) * z_i) # S_T variables
    s_minus = S0 * np.exp((r - σ**2/2) * T - σ * np.sqrt(T) * z_i)
    
    if corr_display:
        corr = np.corrcoef( np.where(np.logical_and(a < s_plus , s_plus < b), 1, 0), 
                            np.where(np.logical_and(a < s_minus , s_minus < b), 1, 0))[0,1]
        
        print('Correlation between the 1[a < S_i(+) < b] and 1[a < S_i(+) < b] random variables: {:.4f}'.format(corr))

    MC_samples = ( K * np.exp(-r * T)) * ( np.where(np.logical_and(a < s_plus , s_plus < b), 1, 0) 
                                          + np.where(np.logical_and(a < s_minus , s_minus < b), 1, 0) ) / 2
    
    antithetic_estimate_price = MC_samples.mean()
    
    antithetic_estimate_var = MC_samples.var() / len(MC_samples)
    
    return antithetic_estimate_price, antithetic_estimate_var



def Euler_simulate_S(h, T, r, σ, S0, seed = 1698, return_path = False):
    
    """
    Parameters
    ----------
    h: Time interval between successive grid points.
    r: The non-negative constant interest rate.
    T: Maturity date (must be positive).
    σ: Volatility of the stock (must be positive).
    S0: Initial stock price (must be positive).
    seed: Seed for random number generation reproducibility.
    return_path: Set this to True if you want the function to return the whole path of the prices.
    
    Returns
    -------
    S : Returns a maturity stock price realisation (using return_path=False).
    z : Returns the sample of standard normals used in the scheme of part (a) (useful for subsequent parts of the question).
    """
    
    n_steps = int(T / h) # Number of points in the grid
    rng = np.random.default_rng(seed) # Set the seed for reproducible results
    z = rng.standard_normal(n_steps) 
    
    S = [S0] # Initialise a list to store the approximation values of S
    
    for i in range(n_steps - 1):
        
        s = S[i] * ( 1 + r * h ) + ( σ * np.sqrt(S[i]) ) * ( np.sqrt(h) * z[i+1]  ) # S_(i+1)h estimate using part (a)
        
        # Make sure that S values don't go below zero
        if s < 0:
            S.append(0)
        else:
            S.append(s)
            
    if return_path: # return the whole path
        return S , z
    
    return S[-1] , z   




def Q4_standard_MC_option_pricing(nsamples, c_level, h, K, r, T, σ, S0, a, b):
    
    """
    Parameters
    ----------
    nsamples: Number of simulated random variables to be used in the Monte Carlo calculations
    c_level: Level of the confidence interval, should be in (0,1)
    h: Time interval between successive grid points, used in the Euler scheme
    K: Fixed payoff of the option when it is in-the-money (must be positive)
    r: The non-negative constant interest rate
    T: Maturity date (must be positive)
    σ: Volatility of the stock (must be positive)
    S0: Initial stock price (must be positive)
    a: Lower end-point of the in-the-money interval of the option (must be positive)
    b: Upper end-point of the in-the-money interval of the option (must be greater than a)
    
    Returns
    -------
    MC_option_price : Monte Carlo estimate for the time-0 option price
    MC_option_price_var: Variance of the standard Monte Carlo estimate for the time-0 option price
    confidence_interval: Asymptotic ('c_level' * 100)% confidence interval for the time-0 option price 
    
    """
    
    alpha = sc.stats.norm.ppf(q = 1 - ( (1 - c_level) / 2) , loc = 0, scale = 1) # Used in the confidence interval
    
    S_T = [] # initialise a list to store the generated maturity stock prices
    
    for i in range(nsamples):
        S_T.append(Euler_simulate_S(h, T, r, σ, S0, seed = 100 * i, return_path = False)[0]) # store maturity price
    
    
    # MC estimator for Q[ a < S_T < b ] - This part is the same as in the MC estimator in Problem 3(c.2)
    
    I_MC = sum( np.logical_and( a < np.array(S_T) , np.array(S_T) < b) ) / nsamples
    
    I_MC_std = ( (I_MC * (1 - I_MC)) / nsamples ) ** 0.5 # standard deviation of I_MC
    
    MC_option_price = (K * np.exp(-r * T)) * I_MC # MC estimate for the option price
     
    MC_option_price_std = (K * np.exp(-r * T)) * I_MC_std # standard deviation of option price MC estimate
    
    MC_option_price_var = MC_option_price_std**2 # variance of option price MC estimate
    
    confidence_interval = [  ( MC_option_price - (alpha * MC_option_price_std) ) , 
                               ( MC_option_price + (alpha * MC_option_price_std) ) ]
    

    return MC_option_price , MC_option_price_var, confidence_interval




def Q4_control_variates_pricing(control_type, nsamples, h, K, r, T, σ, S0, a, b, corr_display = False):
    
    """
    Parameters
    ----------
    control_type: Use this argument to choose the random variable used as control, as described above (enter either 1 or 2).
    nsamples: Number of simulated random variables to be used in the Monte Carlo calculations.
    h: Time interval between successive grid points, used in the Euler scheme
    K: Fixed payoff of the option when it is in-the-money (must be positive).
    r: The non-negative constant interest rate.
    T: Maturity date (must be positive).
    σ: Volatility of the stock (must be positive).
    S0: Initial stock price (must be positive).
    a: Lower end-point of the in-the-money interval of the option (must be positive).
    b: Upper end-point of the in-the-money interval of the option (must be greater than a).
    corr_display: Set this to True if you want the function to print the sample correlation between this Xi s and Yi s.
    
    Returns
    -------
    cv_estimator_price : Control variates estimate for the time-0 option price.
    cv_estimator_var: Variance of the Control variates estimate for the time-0 option price.
    
    """
    
    S_T = np.zeros(nsamples) # initialise an array to store the generated maturity stock prices
    
    Z_i = np.zeros(nsamples) # store the *SUM* of standard normals from each path realisation
    
    for k in range(nsamples):
        
        ST , Z = (Euler_simulate_S(h, T, r, σ, S0, seed = 100 * k, return_path = False )) 
        S_T[k] = ST
        Z_i[k] = sum(Z) 
    
    if control_type == 1: # Use control variate estimator 1 from above (stock price used as a control)
        
        X_i = np.exp(-r * T) * S_T
        Y_i = ( K * np.exp(-r * T) ) * np.where(np.logical_and(a < np.exp(r * T) * X_i , np.exp(r * T) * X_i < b), 1, 0)
        cv_correction = X_i - S0 # correction term in Y_i(b)
        bstar = np.cov(Y_i, cv_correction)[0, 1] / cv_correction.var()
        MC_samples = Y_i - bstar * cv_correction
        
        if corr_display:
            print(f'Correlation between X_i s and Y_i s: {np.corrcoef(X_i, Y_i)[0,1] : .4f}')
    
    else: # Use control variate estimator 2 from above
        
        Y_i = K * np.exp(-r * T) * np.where(np.logical_and(a < S_T , S_T < b), 1, 0)
        bstar = np.cov(Y_i, Z_i)[0, 1] / Z_i.var()
        MC_samples = Y_i - bstar * Z_i
        
        if corr_display:
            print(f'Correlation between X_i s and Y_i s: {np.corrcoef(Z_i, Y_i)[0,1] : .4f}')
        
    cv_estimator_price = MC_samples.mean() # Estimate for the time-0 option price
    
    cv_estimator_std = MC_samples.std() / np.sqrt(len(MC_samples))
    
    cv_estimator_var = cv_estimator_std ** 2
    
    
    return cv_estimator_price , cv_estimator_var
