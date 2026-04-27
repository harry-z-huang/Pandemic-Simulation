import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import inv_boxcox
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()

def Gillespie_model(P,I0,beta,gamma, days,mu,V0):

    S=P-I0-V0
    I =I0
    R=V0
    t=0
    D=0

    t_=[]
    s_=[]
    i_=[]
    r_=[]
    d_=[]
    R0 = beta/gamma
    HT = 1-1/R0 # heard immunity threshold

    while t< days and I>0:
        rate_infection = S*(beta*I/P) # no. suceptible*prob. of infection
        rate_recovery = gamma * I
        death_rate = mu*I
        total_rate = rate_infection + rate_recovery + death_rate
        dt = np.random.exponential(1 / total_rate) # events have exponential distribution
        t = t+dt

        infection_prob = rate_infection/total_rate
        recovery_prob = rate_recovery/total_rate



        if total_rate ==0:
            break
        RN = np.random.random()
        if RN < infection_prob:
            #infection
            S =S-1  
            I =I+1
        #recovery
        elif RN < infection_prob + recovery_prob:
            I =I-1
            R =R+1
        else: #deaths
            I=I-1
            D=D+1
        t_.append(t)
        s_.append(S)
        i_.append(I)
        r_.append(R)
        d_.append(D)

    return t_,s_,i_,r_,d_

## Repeat the simulation for mutiple times 
# first define the initial values

def multi_run(P,V0,I0,beta,gamma,days,n_runs,mu):
    T_=[]
    S_=[]
    I_=[]
    R_=[]
    D_=[]
    R0 = beta/gamma
    HT = 1-1/R0 # heard immunity threshold
    np.random.seed(42)

    for i in range(0,n_runs):
        t_,s_,i_,r_,d_ = Gillespie_model(P,I0,beta,gamma, days,mu,V0)
        T_.append(t_)
        S_.append(s_)
        I_.append(i_)
        R_.append(r_)
        D_.append(d_)

    return T_,S_,I_,R_,D_

def goodness_of_fit(data, f, args): ##Kolmogorov-Smirnov test 
    ks_stat, p_value = stats.kstest(data, f, args =args)
    IsgoodFit='no'
    if p_value > 0.05:
        IsgoodFit='yes'   
    return round(ks_stat,3),round(p_value,3)

def confidence_int(data,mu, sigma,Z):
    n=len(data)
    fitted_mean = np.exp(mu + sigma**2 / 2)
    se_mu    = sigma / np.sqrt(n)
    se_sigma = sigma / np.sqrt(2 * n)
    z=Z
    lower = np.exp((mu - z * se_mu) + (sigma - z * se_sigma)**2 / 2)
    upper = np.exp((mu + z * se_mu) + (sigma + z * se_sigma)**2 / 2)
    return lower,upper


def make_dataframe(T_,S_,I_,R_, D_, P,mu,cutoff):
    min_s=[np.min(lst) for lst in S_] #not infected
    max_i=[np.max(lst) for lst in I_] #max infected
    max_index =  [np.argmax(lst) for lst in I_]
    deaths = [np.max(lst) for lst in D_]

    peak_time =[]
    for i in range(len(T_)):
        peak_time.append(T_[i][max_index[i]])

    df = pd.DataFrame()
    df['not_infected'] = min_s
    df['max_infected'] = max_i
    df['Recovered'] = [np.max(lst) for lst in R_]
    df['Peaktime'] = peak_time
    df['deaths'] = deaths
    df = df[df['deaths']>= P*mu/10]
    # cutoff = df['Peaktime'].sort_values()[0:15].max()
    # print(cutoff)
    df = df[df['Peaktime']>=cutoff]
    df = df.dropna()
    # print(mu*P/10)
    return df


def get_stat_peaktime(df, output_name=None):

    data = df['Peaktime'].dropna().values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.hist(df['Peaktime'],bins=25,label='Peak time')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Counts/Bin')
    ax1.set_title('Peak time')
    #ax1.legend()
    ax1.grid(True, alpha=0.3)

    tr_data, lambda_ = stats.boxcox(data)

    sns.histplot(tr_data, kde=True, ax=ax2, color='blue')
    ax2.set_title(f"Box-Cox Transformed (λ = {lambda_:.2f})")
    ax2.set_ylabel('Counts/Bin')
    if output_name:
        save_figure(output_name)

    #mean and CI
    mean = np.mean(tr_data)
    sigma = np.std(tr_data, ddof=1) 

    # 2. 95% CI
    lower, higher = stats.t.interval(0.95, 
                                   df=len(tr_data)-1, 
                                   loc=mean, 
                                   scale=stats.sem(tr_data))

    mean_ = inv_boxcox(mean, lambda_)
    lower_ = inv_boxcox(lower, lambda_)
    higher_ = inv_boxcox(higher, lambda_)


    # Goodness of fit
    shapiro_stat, p_value = stats.shapiro(tr_data)

    return mean_,lower_,higher_, shapiro_stat,p_value

def get_stat_maxInfections(df, output_name=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    df1 = df[df['max_infected']>20]
    data = df1['max_infected'].dropna().values

    ax1.hist(data,bins=25,label='')
    ax1.set_xlabel('# of people infected')
    ax1.set_ylabel('Counts/Bin')
    ax1.set_title('Infections at the peak')
    #ax1.legend()
    ax1.grid(True, alpha=0.3)

    tr_data, lambda_ = stats.boxcox(data)

    sns.histplot(tr_data, kde=True, ax=ax2, color='blue')
    ax2.set_title(f"Box-Cox Transformed (λ = {lambda_:.2f})")
    ax2.set_ylabel('Counts/Bin')
    if output_name:
        save_figure(output_name)

    #mean and CI
    mean = np.mean(tr_data)
    sigma = np.std(tr_data, ddof=1) 

    # 2. 95% CI
    lower, higher = stats.t.interval(0.95, 
                                   df=len(tr_data)-1, 
                                   loc=mean, 
                                   scale=stats.sem(tr_data))

    mean_ = inv_boxcox(mean, lambda_)
    lower_ = inv_boxcox(lower, lambda_)
    higher_ = inv_boxcox(higher, lambda_)

    shapiro_stat, p_value = stats.shapiro(tr_data)
    return mean_,lower_,higher_,shapiro_stat,p_value

def get_deaths(D_, output_name=None):
    deaths=[np.max(lst) for lst in D_]
    data = pd.DataFrame()
    data['deaths'] = deaths
    data = data[data['deaths']>200]
    data = data['deaths'].dropna().values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.hist(data,bins=25,label='')
    ax1.set_xlabel('# of people died')
    ax1.set_ylabel('Counts/Bin')
    ax1.set_title('# people died of Infection')
    ax1.grid(True, alpha=0.3)

    tr_data, lambda_ = stats.boxcox(data)

    sns.histplot(tr_data, kde=True, ax=ax2, color='blue')
    ax2.set_title(f"Box-Cox Transformed (λ = {lambda_:.2f})")
    ax2.set_ylabel('Counts/Bin')
    if output_name:
        save_figure(output_name)

    mean = np.mean(tr_data)
    sigma = np.std(tr_data, ddof=1) 

    # 2. 95% CI
    lower, higher = stats.t.interval(0.95, 
                                   df=len(tr_data)-1, 
                                   loc=mean, 
                                   scale=stats.sem(tr_data))

    mean_ = inv_boxcox(mean, lambda_)
    lower_ = inv_boxcox(lower, lambda_)
    higher_ = inv_boxcox(higher, lambda_)

    print(f"Mean Deaths: {mean_:.2f}")
    print(f"95% CI : ({lower_:.2f}, {higher_:.2f})")
    shapiro_stat, p_value = stats.shapiro(tr_data)

    print(f"Shapiro-Wilk Statistic: {shapiro_stat:.2f}")
    print(f"P-value: {p_value:.2f}")

def get_recovered(R_,V0,cutoff, output_name=None):
    recovered=[np.max(lst) for lst in R_]
    data = pd.DataFrame()
    data['recovered'] = recovered 
    data['recovered'] = data['recovered']-V0 #subtract vaccinations
    data = data[data['recovered']>cutoff]
    data = data['recovered'].dropna().values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.hist(data,bins=25,label='')
    ax1.set_xlabel('# of people recovered')
    ax1.set_ylabel('Counts/Bin')
    ax1.set_title('# people Recovered')
    ax1.grid(True, alpha=0.3)

    tr_data, lambda_ = stats.boxcox(data)

    sns.histplot(tr_data, kde=True, ax=ax2, color='blue')
    ax2.set_title(f"Box-Cox Transformed (λ = {lambda_:.2f})")
    ax2.set_ylabel('Counts/Bin')
    if output_name:
        save_figure(output_name)

    mean = np.mean(tr_data)
    sigma = np.std(tr_data, ddof=1) 
        # 2. 95% CI
    lower, higher = stats.t.interval(0.95, 
                                   df=len(tr_data)-1, 
                                   loc=mean, 
                                   scale=stats.sem(tr_data))

    mean_ = inv_boxcox(mean, lambda_)
    lower_ = inv_boxcox(lower, lambda_)
    higher_ = inv_boxcox(higher, lambda_)

    print(f"Mean- Recovered : {mean_:.2f}")
    print(f"95% CI : ({lower_:.2f}, {higher_:.2f})")
    shapiro_stat, p_value = stats.shapiro(tr_data)

    print(f"Shapiro-Wilk Statistic: {shapiro_stat:.2f}")
    print(f"P-value: {p_value:.2f}")
# ### 1.0 Simulation - with no Intervention
# #### Total population: 100,000
# #### Initial infections : 10
# #### vaccinations: 0
# #### death rate =0

P = 100000 # total population
V0 =0 # vaccinated
I0 = 1 # infected
beta =0.3
gamma =0.2
days =200
mu =0.0 # death rate

np.random.seed(42)
a,b,c,d,e =Gillespie_model(100000,10,0.3,0.2, 200,0,V0=0)
R0 = beta/gamma
HT = (1-1/R0)*P # heard immunity threshold
plt.plot(a,c,label='Infectious')
plt.plot(a,b,label='Susceptible')
plt.plot(a,d,label='Recovered')

plt.axhline(y=HT, color='brown', linestyle='--',label='Herd Immunity Threshold')
peak =a[np.argmax(c)]
plt.axvline(x=peak, color='pink', linestyle='--',label='Peak')
plt.xlabel("Days passed")
plt.ylabel("No. of people")
plt.legend()
save_figure('scenario_1_no_intervention_timeseries.png')

## Peak time
T_,S_,I_,R_,D_ =multi_run(100000,0,I0=10,beta=0.3,gamma=0.2,days=200,n_runs=200, mu=0)
df = make_dataframe(T_,S_,I_,R_, D_, 100000,0,cutoff=50)
mean,lower_,higher_, shapiro_stat,p_value = get_stat_peaktime(df, 'scenario_1_peak_time.png')

print(f'Mean days to peak of the pandemic : {mean:.2f}')
print(f'95% confidence interval for the mean : [{lower_:.2f},{higher_:.2f}]')
print(f'Goodness of fit results (Shapiro-Wilk) : shapiro_stat={shapiro_stat:.2f}, p_value={p_value:.2f}')

# df['Peaktime'].sort_values()[0:10]

##No of infections at the peak
mean,lower,upper,shapiro_stat,p_value = get_stat_maxInfections(df, 'scenario_1_peak_infections.png')
print(f'Mean of maximun infections during the pandemic : {mean:.2f}')
print(f'95% confidence interval for the mean : [{lower:.2f},{upper:.2f}]')
print(f'Goodness of fit results (Shapiro-Wilk) : k_stat={shapiro_stat:.2f}, p_value={p_value:.2f}')

### no. of recovered (same as # gor sick in this case)
get_recovered(R_,0,18000, 'scenario_1_recovered.png')

# ### 1.1 Simuation with deaths
# #### death rate = 0.01

np.random.seed(42)
a,b,c,d,e =Gillespie_model(100000,10,0.3,0.2, 200,0.01,V0=0)
R0 = beta/gamma
HT = (1-1/R0)*P # heard immunity threshold
plt.plot(a,c,label='Infectious')
plt.plot(a,b,label='Susceptible')
plt.plot(a,d,label='Recovered')
plt.plot(a,e,label='Deaths')
plt.axhline(y=HT, color='brown', linestyle='--',label='Herd Immunity Threshold')
peak =a[np.argmax(c)]
plt.axvline(x=peak, color='pink', linestyle='--',label='Peak')
plt.xlabel("Days passed")
plt.ylabel("No. of people")
plt.legend()
save_figure('scenario_2_deaths_timeseries.png')

## no intervention but 0.01 death rate
T_,S_,I_,R_,D_ =multi_run(100000,0,I0=10,beta=0.3,gamma=0.2,days=200,n_runs=200, mu=0.01)
df = make_dataframe(T_,S_,I_,R_, D_, 100000,0.01,cutoff=0)
mean,lower_,higher_, shapiro_stat,p_value = get_stat_peaktime(df, 'scenario_2_peak_time.png')

print(f'Mean days to peak of the pandemic : {mean:.2f}')
print(f'95% confidence interval for the mean : [{lower_:.2f},{higher_:.2f}]')
print(f'Goodness of fit results (Shapiro-Wilk) : shapiro_stat={shapiro_stat:.2f}, p_value={p_value:.2f}')

mean,lower,upper,shapiro_stat,p_value = get_stat_maxInfections(df, 'scenario_2_peak_infections.png')
print(f'Mean of maximun infections during the pandemic : {mean:.2f}')
print(f'95% confidence interval for the mean : [{lower:.2f},{upper:.2f}]')
print(f'Goodness of fit results (Shapiro-Wilk) : k_stat={shapiro_stat:.2f}, p_value={p_value:.2f}')

get_deaths(D_, 'scenario_2_deaths_distribution.png')

get_recovered(R_,0,18000, 'scenario_2_recovered.png')


# ### 2 Simuation with deaths + vaccinations
# #### death rate = 0.01
# #### vaccinations : 10%
# ### V0=P*0.1

np.random.seed(42)
V0 = P*0.1
a,b,c,d,e =Gillespie_model(100000,10,0.3,0.2, 200,0.01,V0=P*0.01)
R0 = beta/gamma
HT = (1-1/R0)*P # heard immunity threshold
plt.plot(a,c,label='Infectious')
plt.plot(a,b,label='Susceptible')
plt.plot(a,d,label='Recovered')
plt.plot(a,e,label='Deaths')
plt.axhline(y=HT, color='brown', linestyle='--',label='Herd Immunity Threshold')
peak =a[np.argmax(c)]
plt.axvline(x=peak, color='pink', linestyle='--',label='Peak')
plt.xlabel("Days passed")
plt.ylabel("No. of people")
plt.legend()
save_figure('scenario_3_vaccination_timeseries.png')

T_,S_,I_,R_,D_ =multi_run(100000,V0=P*0.01,I0=10,beta=0.3,gamma=0.2,days=200,n_runs=200, mu=0.01)

df = make_dataframe(T_,S_,I_,R_, D_, 100000,0.01,0)
mean,lower_,higher_, shapiro_stat,p_value = get_stat_peaktime(df, 'scenario_3_peak_time.png')

print(f'Mean days to peak of the pandemic : {mean:.2f}')
print(f'95% confidence interval for the mean : [{lower_:.2f},{higher_:.2f}]')
print(f'Goodness of fit results (Shapiro-Wilk) : shapiro_stat={shapiro_stat:.2f}, p_value={p_value:.2f}')

mean,lower,upper,shapiro_stat,p_value = get_stat_maxInfections(df, 'scenario_3_peak_infections.png')
print(f'Mean of maximun infections during the pandemic : {mean:.2f}')
print(f'95% confidence interval for the mean : [{lower:.2f},{upper:.2f}]')
print(f'Goodness of fit results (Shapiro-Wilk) : k_stat={shapiro_stat:.2f}, p_value={p_value:.2f}')

get_deaths(D_, 'scenario_3_deaths_distribution.png')

get_recovered(R_,P*0.01,30000, 'scenario_3_recovered.png')

#https://pmc.ncbi.nlm.nih.gov/articles/PMC8119989/
#https://pmc.ncbi.nlm.nih.gov/articles/PMC8063609/
#https://onlinelibrary.wiley.com/doi/full/10.1002/mma.7965


# ### 3 Simuation with deaths + vaccinations + Social distancing
# #### death rate = 0.01
# #### vaccinations : 10%
# ### V0=P*0.1
# ### Social ditancing: beta=0.27

np.random.seed(42)
V0 = P*0.1
a,b,c,d,e =Gillespie_model(100000,10,0.27,0.2, 200,0.01,V0=P*0.01)
R0 = beta/gamma
HT = (1-1/R0)*P # heard immunity threshold
plt.plot(a,c,label='Infectious')
plt.plot(a,b,label='Susceptible')
plt.plot(a,d,label='Recovered')
plt.plot(a,e,label='Deaths')
plt.axhline(y=HT, color='brown', linestyle='--',label='Herd Immunity Threshold')
peak =a[np.argmax(c)]
plt.axvline(x=peak, color='pink', linestyle='--',label='Peak')
plt.xlabel("Days passed")
plt.ylabel("No. of people")
plt.legend()
save_figure('scenario_4_distancing_timeseries.png')

T_,S_,I_,R_,D_ =multi_run(100000,V0=P*0.01,I0=10,beta=0.27,gamma=0.2,days=200,n_runs=200, mu=0.01)

df = make_dataframe(T_,S_,I_,R_, D_, 100000,0.01,0)
mean,lower_,higher_, shapiro_stat,p_value = get_stat_peaktime(df, 'scenario_4_peak_time.png')

print(f'Mean days to peak of the pandemic : {mean:.2f}')
print(f'95% confidence interval for the mean : [{lower_:.2f},{higher_:.2f}]')
print(f'Goodness of fit results (Shapiro-Wilk) : shapiro_stat={shapiro_stat:.2f}, p_value={p_value:.2f}')

mean,lower,upper,shapiro_stat,p_value = get_stat_maxInfections(df, 'scenario_4_peak_infections.png')
print(f'Mean of maximun infections during the pandemic : {mean:.2f}')
print(f'95% confidence interval for the mean : [{lower:.2f},{upper:.2f}]')
print(f'Goodness of fit results (Shapiro-Wilk) : k_stat={shapiro_stat:.2f}, p_value={p_value:.2f}')

get_deaths(D_, 'scenario_4_deaths_distribution.png')

get_recovered(R_,P*0.01,4000, 'scenario_4_recovered.png')

# df['Recovered']
# df['Recovered']-V0




