#%%
import numpy as np
from scipy.linalg import svd
from scipy import stats
import matplotlib.pyplot as plt
from math import sqrt,log,floor
T=20000


m=50
n=50
r=2
pr=1-1/n
context_distribution=[1/m for i in range(m)]


#%%
#generate ground-truth matrix and its SVD 

N = np.random.rand(n, n)
Nx = np.sum(np.abs(N), axis=1)
np.fill_diagonal(N,Nx)

M=np.random.rand(m,m)
Mx = np.sum(np.abs(M), axis=1)
np.fill_diagonal(M,Mx)
Ir=np.zeros((m,n))
for i in range (r):
    Ir[i][i]=1
true_matrix=M.dot(Ir).dot(N)

best_arms=[1+np.argmax(true_matrix[x]) for x in range(m)]

def rank_r_svd(matrix):
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    U, s, VT = svd(matrix)
    truncU=U[:,0:r]
    orthU=U[:,r:m]
    truncsigma = np.zeros((r,r))
    truncsigma[:r, :r] = np.diag(s[0:r])
    truncVT=VT[0:r,:]
    orthVT=VT[r:n,:]
    return (truncU,orthU,truncsigma,truncVT,orthVT)

true_matrix_svd=rank_r_svd(true_matrix)


#%%
#define target and behavior policies

def behavior(p,n,x,a): #take p=1-1/n for uniform
    if a==1:
        return 1-p
    else:
        return p/(n-1)

def target(x,a):
    if a==best_arms[x-1]:  #if there's more than one argmax, outputs the first argmax.
        return 1
    else :
        return 0

#%%
#generate samples and estimated rank-r matrix
        
def samples_generation(T,pr,matrix):
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    samples=[]
    for t in range(T):
        x=np.random.choice(np.arange(1,m+1), p=[1/m for i in range(m)])
        a=np.random.choice(np.arange(1,n+1),p=[behavior(pr,n,x,a) for a in range(1,n+1)])
        rw=matrix[x-1][a-1]+np.random.normal(0,1)
        samples.append((x,a,rw))
    return samples
    
def matrix_estimate(samples,nb_samples,pr,m,n):
    matrix=np.zeros((m,n),dtype=float)
    matrix_2=np.zeros((m,n),dtype=float)
    for i in range(nb_samples):
        (x,a)=samples[i][0:2]
        matrix[x-1][a-1]+=samples[i][2]
    for x in range(1,m+1):
        for a in range (1,n+1):
            denominator=(1/m)*behavior(pr,n,x,a)
            matrix_2[x-1][a-1]=matrix[x-1][a-1]/denominator
    return (matrix_2/nb_samples)

def rank_r_estimate(samples,nb_samples,pr,m,n):  
    estimated_matrix=matrix_estimate(samples,nb_samples,pr,m,n)
    (truncU,orthU,truncsigma,truncVT,orthVT)=rank_r_svd(estimated_matrix)
    return truncU.dot(truncsigma.dot(truncVT))

#%%
#define feature maps and other useful quantities

def phi(x,a,matrix_svd):
    (truncU,orthU,truncsigma,truncVT,orthVT)=matrix_svd
    col_U=truncU[x-1:x,:]
    col_V=truncVT[:,a-1:a]
    col_orthU=orthU[x-1:x,:]
    col_orthV=orthVT[:,a-1:a]
    matrix_1=np.dot(col_V,col_U)
    vec_1=matrix_1.flatten()
    matrix_2=np.dot(col_orthV,col_U)
    vec_2=matrix_2.flatten()
    matrix_3=np.dot(col_V,col_orthU)
    vec_3=matrix_3.flatten()
    return np.concatenate((vec_1,vec_2,vec_3))

def B(pr,matrix_svd):
    m=np.size(matrix_svd[0],axis=0)
    n=np.size(matrix_svd[3],axis=1)
    matrix_sum=(1/m)*sum(behavior(pr,n,x,a)*np.dot(phi(x,a,matrix_svd)[:, np.newaxis],phi(x,a,matrix_svd).T[np.newaxis, :]) for x in range(1,m+1) for a in range(1,n+1))   
    return matrix_sum

def v(matrix_svd):
    m=np.size(matrix_svd[0],axis=0)
    n=np.size(matrix_svd[3],axis=1)
    vector_sum=(1/m)*sum(target(x,a)*phi(x,a,matrix_svd) for x in range(1,m+1) for a in range(1,n+1))
    return vector_sum
            
def instance_upper_bound(pr,matrix_svd):
    inv_B=np.linalg.inv(B(pr,matrix_svd)) 
    v_pi=v(matrix_svd)
    return  np.sqrt(v_pi.dot(inv_B.dot(v_pi)))

def bernoulli_kl(delta):
    return delta*log(delta/(1-delta))+(1-delta)*log((1-delta)/delta)

def instance_lower_bound(pr,matrix):
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    (truncU,orthU,truncsigma,truncVT,orthVT)=rank_r_svd(matrix)
    psi_1=truncU.dot(np.sqrt(truncsigma))
    psi_2=np.transpose(truncVT).dot(np.sqrt(truncsigma))
    first_term,second_term=0,0
    for x in range(1,m+1):
            v_x=sum((1/m)*target(x,a)*psi_2[a-1] for a in range(1,n+1))
            B_x=sum((1/m)*behavior(pr,n,x,a)*np.dot(psi_2[a-1][:, np.newaxis],psi_2[a-1].T[np.newaxis, :]) for a in range (1,n+1))
            first_term+=v_x.dot(np.linalg.inv(B_x).dot(v_x))
    for a in range (1,n+1):
            v_a=sum((1/m)*target(x,a)*psi_1[x-1] for x in range(1,m+1))
            B_a=sum((1/m)*behavior(pr,n,x,a)*np.dot(psi_1[x-1][:, np.newaxis],psi_1[x-1].T[np.newaxis, :]) for x in range (1,m+1))
            second_term+=v_a.dot(np.linalg.inv(B_a).dot(v_a))
    return np.sqrt(max(first_term,second_term))

def policy_value(matrix):
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    value=(1/m)*sum(target(x,a)*matrix[x-1][a-1] for a in range (1,n+1) for x in range (1,m+1))
    return value

def best_policy_value(pr,matrix):
    best_policy=[np.argmax(matrix[x]) for x in range(m)]
    policy_value=(1/m)*sum(matrix[x][best_policy[x]] for x in range(m))
    return policy_value

#%%
#estimators

def DM_RS(samples,nb_samples,pr,matrix): #SIPS in the paper
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    estimated_matrix=rank_r_estimate(samples,nb_samples,pr,m,n)  
    error_estimate=(1/m)*sum(target(x,a)*(matrix[x-1][a-1]-estimated_matrix[x-1][a-1]) for a in range (1,n+1) for x in range (1,m+1))
    return np.abs(error_estimate)  
  
def DM_RS_MLB_split(samples,nb_samples,pr,regu,samples_proportion,matrix): #RS-PE in the paper
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    T_1=floor(samples_proportion*nb_samples)
    estimated_matrix=matrix_estimate(samples,T_1,pr,m,n)
    estimated_svd=rank_r_svd(estimated_matrix)
    feature=[[phi(x,a,estimated_svd) for a in range(1,n+1)] for x in range(1,m+1)] #building the feature map components
    v_hat=(1/m)*sum(target(x,a)*feature[x-1][a-1] for a in range(1,n+1) for x in range(1,m+1)) #building the target-feature vector
    cov_matrix=0 #building the least square estimator
    reward_vector=0
    for (x,a,rw) in samples[T_1:]:
       cov_matrix+=np.dot(feature[x-1][a-1][:, np.newaxis],feature[x-1][a-1].T[np.newaxis, :])
       reward_vector+=rw*feature[x-1][a-1]
    d=np.size(cov_matrix,axis=0)
    lse=np.linalg.inv(regu*np.identity(d)+cov_matrix).dot(reward_vector)
    policy_estimate=v_hat.dot(lse)
    return np.abs(policy_estimate-policy_value(matrix))

def DM_RS_MLB(samples,nb_samples,pr,regu,matrix): #RS-PE, no data splitting
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    estimated_matrix=matrix_estimate(samples,nb_samples,pr,m,n)
    estimated_svd=rank_r_svd(estimated_matrix)
    feature=[[phi(x,a,estimated_svd) for a in range(1,n+1)] for x in range(1,m+1)] #building the feature map components
    v_hat=0 #building the target-feature vector
    for x in range(1,m+1):
        for a in range(1,n+1):
            v_hat+=(1/m)*target(x,a)*feature[x-1][a-1]
    cov_matrix=0 #building the least square estimator
    reward_vector=0
    for (x,a,rw) in samples:
       cov_matrix+=np.dot(feature[x-1][a-1][:, np.newaxis],feature[x-1][a-1].T[np.newaxis, :])
       reward_vector+=rw*feature[x-1][a-1]
    d=np.size(cov_matrix,axis=0)
    lse=np.linalg.inv(regu*np.identity(d)+cov_matrix).dot(reward_vector)
    policy_estimate=v_hat.dot(lse)
    return np.abs(policy_estimate-policy_value(matrix))

def reward_estimate(samples,nb_samples,pr,m,n): #no data splitting   
    estimated_matrix=matrix_estimate(samples,nb_samples,pr,m,n)
    estimated_svd=rank_r_svd(estimated_matrix)
    feature=[[phi(x,a,estimated_svd) for a in range(1,n+1)] for x in range(1,m+1)] #building the feature map components
    cov_matrix=0 #building the least square estimator
    reward_vector=0
    for (x,a,rw) in samples:
       cov_matrix+=np.dot(feature[x-1][a-1][:, np.newaxis],feature[x-1][a-1].T[np.newaxis, :])
       reward_vector+=rw*feature[x-1][a-1]
    d=np.size(cov_matrix,axis=0)
    lse=np.linalg.inv(0.00001*np.identity(d)+cov_matrix).dot(reward_vector)
    reward_estimate=np.zeros((m,n))
    for x in range(1,m+1):
        for a in range(1,n+1):
            reward_estimate[x-1][a-1]=feature[x-1][a-1].T.dot(lse)
    return reward_estimate
    
    
def IPS(samples,nb_samples,pr,matrix):
    n=np.size(matrix,axis=1)
    value_estimate=0
    for i in range(nb_samples):
        (x,a,rw)=samples[i]
        ips_term=(target(x,a)/behavior(pr,n,x,a))*rw
        value_estimate+=ips_term
    value_estimate/=nb_samples
    return np.abs(value_estimate-policy_value(matrix))

def RS_BPI(samples,nb_samples,pr,matrix): #SBPI in the paper
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    estimated_matrix=rank_r_estimate(samples,nb_samples,pr,m,n)
    policy_estimate=[0 for x in range(m)]
    for x in range(m):
        policy_estimate[x]=np.argmax(estimated_matrix[x])
    policy_value=(1/m)*sum(matrix[x][policy_estimate[x]] for x in range(m))
    return policy_value

def RS_MLB_BPI(samples,nb_samples,pr,matrix): #RS-BPI in the paper
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    estimated_matrix=reward_estimate(samples,nb_samples,pr,m,n)
    policy_estimate=[0 for x in range(m)]
    for x in range(m):
        policy_estimate[x]=np.argmax(estimated_matrix[x])
    policy_value=(1/m)*sum(matrix[x][policy_estimate[x]] for x in range(m))
    return policy_value

def IPS_BPI(samples,nb_samples,pr,matrix): #'Benchmark' in the paper
    m=np.size(matrix,axis=0)
    n=np.size(matrix,axis=1)
    estimated_matrix=matrix_estimate(samples,nb_samples,pr,m,n)
    policy_estimate=[0 for x in range(m)]
    for x in range(m):
        policy_estimate[x]=np.argmax(estimated_matrix[x])
    policy_value=(1/m)*sum(matrix[x][policy_estimate[x]] for x in range(m))
    return policy_value


 
#%% Generating the data

samples=samples_generation(T,pr,true_matrix)

#%%
#Experiment 1: impact of data splitting. Uncomment to run.
# nb_experiments=50
# pr=1-1/n
# upper_bound=instance_upper_bound(pr,true_matrix_svd)
# lower_bound=instance_lower_bound(pr,true_matrix)
# x_range=(np.arange(4,sqrt(T),3)**2).astype(int)
# policy_error=[]
# policy_error2=[]
# policy_error3=[]
# policy_error4=[]
# for i in range(nb_experiments):
#     samples=samples_generation(T,pr,true_matrix)
#     policy_error.append([DM_RS_MLB(samples,x,pr,0.0001,true_matrix) for x in x_range])
#     policy_error2.append([DM_RS_MLB_split(samples,x,pr,0.0001,1/2,true_matrix) for x in x_range])
#     policy_error3.append([DM_RS_MLB_split(samples,x,pr,0.0001,1/5,true_matrix) for x in x_range])
#     policy_error4.append([DM_RS_MLB_split(samples,x,pr,0.0001,4/5,true_matrix) for x in x_range])

# beta=stats.t(df=49).ppf(0.95)/np.sqrt(49)
# mean_error=np.mean(policy_error,axis=0)
# std_error=np.std(policy_error,axis=0)
# mean_error2=np.mean(policy_error2,axis=0)
# std_error2=np.std(policy_error2,axis=0)
# mean_error3=np.mean(policy_error3,axis=0)
# std_error3=np.std(policy_error3,axis=0)
# mean_error4=np.mean(policy_error4,axis=0)
# std_error4=np.std(policy_error4,axis=0)
# beta=stats.t(df=nb_experiments-1).ppf(0.95)/np.sqrt(nb_experiments-1)
# plt.figure()

# plt.yscale("log")
# plt.plot(np.sqrt(x_range),mean_error,marker='|',label='RS-PE',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_error-beta*std_error), (mean_error+beta*std_error), color='b', alpha=.1)
# plt.plot(np.sqrt(x_range),mean_error2, marker='.', label='RS-PE, $α=1/2$',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_error2-beta*std_error2), (mean_error2+beta*std_error2), color='orange', alpha=.1)
# plt.plot(np.sqrt(x_range),mean_error3, marker='x', label='RS-PE, $α=1/5$',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_error3-beta*std_error3), (mean_error3+beta*std_error3), color='green', alpha=.1)
# plt.plot(np.sqrt(x_range),mean_error4, marker='_',label='RS-PE, $α=4/5$',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_error4-beta*std_error4), (mean_error4+beta*std_error4), color='red', alpha=.1)
# plt.xlabel(r'$\sqrt{T}$',fontsize=30)
# plt.ylabel("PE Error",fontsize=30) 
# plt.xlim(4,140)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.legend(loc='best',fontsize=25)
# plt.show()


#%%
#Experiment 2: Impact of regularization. Uncomment to run.
# nb_experiments=50
# pr=1-1/n
# upper_bound=instance_upper_bound(pr,true_matrix_svd)
# lower_bound=instance_lower_bound(pr,true_matrix)
# x_range=(np.arange(4,sqrt(T),3)**2).astype(int)
# policy_error=[]
# policy_error2=[]
# policy_error3=[]
# for i in range(nb_experiments):
#     samples=samples_generation(T,pr,true_matrix)
#     policy_error.append([DM_RS_MLB(samples,x,pr,0.0001,true_matrix) for x in x_range])
#     policy_error2.append([DM_RS_MLB(samples,x,pr,0.01,true_matrix) for x in x_range])
#     policy_error3.append([DM_RS_MLB(samples,x,pr,0.1,true_matrix) for x in x_range])
# beta=stats.t(df=nb_experiments-1).ppf(0.95)/np.sqrt(nb_experiments-1)
# mean_error=np.mean(policy_error,axis=0)
# std_error=np.std(policy_error,axis=0)
# mean_error2=np.mean(policy_error2,axis=0)
# std_error2=np.std(policy_error2,axis=0)
# mean_error3=np.mean(policy_error3,axis=0)
# std_error3=np.std(policy_error3,axis=0)
# # mean_error4=np.mean(policy_error4,axis=0)
# # std_error4=np.std(policy_error4,axis=0)
# plt.figure()

# plt.yscale("log")
# plt.plot(np.sqrt(x_range),mean_error,marker='|',label='RS-PE, $τ=10^{-4}$',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range),(mean_error-beta*std_error), (mean_error+beta*std_error), color='b', alpha=.1)
# plt.plot(np.sqrt(x_range),mean_error2, marker='.', label='RS-PE, $τ=10^{-2}$',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_error2-beta*std_error2), (mean_error2+beta*std_error2), color='orange', alpha=.1)
# plt.plot(np.sqrt(x_range),mean_error3, marker='x', label='RS-PE, $τ=10^{-1}$',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_error3-beta*std_error3), (mean_error3+beta*std_error3), color='green', alpha=.1)
# # plt.plot(np.sqrt(x_range),mean_error4, marker='x', label='RS-PE, $τ=10$',linewidth=3,markersize=12)
# # plt.fill_between(np.sqrt(x_range), (mean_error4-beta*std_error4), (mean_error3+beta*std_error3), color='red', alpha=.1)
# # plt.plot(np.sqrt(x_range),[upper_bound*sqrt(2*log(1600)/x) for x in x_range], linestyle='dashed', label='Asymptotic Upper Bound',linewidth=3) 
# plt.xlabel(r'$\sqrt{T}$',fontsize=30)
# plt.ylabel("PE Error",fontsize=30) 
# plt.xlim(4,140)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.legend(loc='best',fontsize=25)
# plt.show()



#%%
# Experiment 3: comparison between RS-PE,SIPS,IPS. Uncomment to run.
# nb_experiments=50
# pr=1-1/n
# upper_bound=instance_upper_bound(pr,true_matrix_svd)
# lower_bound=instance_lower_bound(pr,true_matrix)
# x_range=(np.arange(4,sqrt(T),3)**2).astype(int)
# policy_error=[]
# policy_error2=[]
# policy_error3=[]
# for i in range(nb_experiments):
#     samples=samples_generation(T,pr,true_matrix)
#     policy_error.append([DM_RS_MLB(samples,x,pr,0.0001,true_matrix) for x in x_range])
#     policy_error2.append([DM_RS(samples,x,pr,true_matrix) for x in x_range]) 
#     policy_error3.append([IPS(samples,x,pr,true_matrix) for x in x_range])
# beta=stats.t(df=nb_experiments-1).ppf(0.95)/np.sqrt(nb_experiments-1)
# mean_error=np.mean(policy_error,axis=0)
# std_error=np.std(policy_error,axis=0)
# mean_error2=np.mean(policy_error2,axis=0)
# std_error2=np.std(policy_error2,axis=0)
# mean_error3=np.mean(policy_error3,axis=0)
# std_error3=np.std(policy_error3,axis=0)
# plt.figure()

# plt.yscale("log")
# plt.plot(np.sqrt(x_range),mean_error,marker='|',label='RS-PE',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_error-beta*std_error), (mean_error+beta*std_error), color='b', alpha=.1)
# plt.plot(np.sqrt(x_range),mean_error2, marker='.', label='SIPS',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_error2-beta*std_error2), (mean_error2+beta*std_error2), color='orange', alpha=.1)
# plt.plot(np.sqrt(x_range),mean_error3, marker='_', label='IPS',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_error3-beta*std_error3), (mean_error3+beta*std_error3), color='green', alpha=.1)
# plt.plot(np.sqrt(x_range),[upper_bound*sqrt(2*log(1600)/x) for x in x_range], linestyle='dashed', label='Asymptotic Upper Bound',linewidth=3) 
# plt.xlabel(r'$\sqrt{T}$',fontsize=30)
# plt.ylabel("PE Error",fontsize=30) 
# plt.xlim(4,140)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.legend(loc='best',fontsize=25)
# plt.show()




#%%
# Experiment 4: Impact of matrix size on the PE error.  Uncomment to run.
# T=10000
# nb_experiments=50
# r=1
# policy_error=[]
# policy_error2=[]
# policy_error3=[]
# size_range=(np.exp(np.arange(1,np.log(320),0.204))).astype(int)
# for i in range(nb_experiments):
#     current_row=[]
#     current_row2=[]
#     current_row3=[]
#     for size in size_range:
#         (m,n)=(size,size)
#         def target(x,a):
#             if a==1:
#                 return 1
#             else: 
#                 return 0
#         pr=1-1/n
#         context_distribution=[1/m for i in range(m)]
#         true_matrix=np.ones((m,n))
#         samples=samples_generation(T,pr,true_matrix)
#         current_row.append(DM_RS_MLB(samples,T,pr,0.0001,true_matrix))
#         current_row2.append(DM_RS(samples,T,pr,true_matrix))
#         current_row3.append(IPS(samples,T,pr,true_matrix))
#     policy_error.append(current_row)
#     policy_error2.append(current_row2)
#     policy_error3.append(current_row3)
# beta=stats.t(df=nb_experiments-1).ppf(0.95)/np.sqrt(nb_experiments-1)
# mean_error=np.mean(policy_error,axis=0)
# std_error=np.std(policy_error,axis=0)
# mean_error2=np.mean(policy_error2,axis=0)
# std_error2=np.std(policy_error2,axis=0)
# mean_error3=np.mean(policy_error3,axis=0)
# std_error3=np.std(policy_error3,axis=0)
# plt.figure()
# plt.xscale("log")
# plt.yscale("log")
# plt.plot(size_range,mean_error,marker='|',label='RS-PE',linewidth=3,markersize=12)
# plt.fill_between(size_range, (mean_error-beta*std_error), (mean_error+beta*std_error), color='b', alpha=.1)
# plt.plot(size_range,mean_error2, marker='.', label='SIPS',linewidth=3,markersize=12)
# plt.fill_between(size_range, (mean_error2-beta*std_error2), (mean_error2+beta*std_error2), color='orange', alpha=.1)
# plt.plot(size_range,mean_error3, marker='_', label='IPS',linewidth=3,markersize=12)
# plt.fill_between(size_range, (mean_error3-beta*std_error3), (mean_error3+beta*std_error3), color='green', alpha=.1)

# plt.xlabel("Matrix Size",fontsize=30)
# plt.xlim(2,296) 
# plt.ylabel("PE Error",fontsize=30) 
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.legend(loc='best',fontsize=25)
# plt.show()


#%%
# Experiment 5: Impact of matrix size on the max-norm.  Uncomment to run.
# T=10000
# nb_experiments=50
# r=1
# good_matrix_error=[]
# bad_matrix_error=[]
# best_matrix_error=[]
# size_range=(np.exp(np.arange(1,np.log(320),0.204))).astype(int)
# for i in range(nb_experiments):
#     current_row=[]
#     current_row2=[]
#     current_row3=[]
#     for size in size_range:
#         (m,n)=(size,size)
#         pr=1-1/n
#         context_distribution=[1/m for i in range(m)]
#         true_matrix=np.ones((m,n))
#         samples=samples_generation(T,pr,true_matrix)
#         good_error_matrix=true_matrix-rank_r_estimate(samples,T,pr,m,n)
#         bad_error_matrix=true_matrix-matrix_estimate(samples,T,pr,m,n)
#         best_error_matrix=true_matrix-reward_estimate(samples,T,pr,m,n)
#         current_row.append(max(np.max(best_error_matrix),-np.min(best_error_matrix)))
#         current_row2.append(max(np.max(good_error_matrix),-np.min(good_error_matrix)))
#         current_row3.append(max(np.max(bad_error_matrix),-np.min(bad_error_matrix)))
#     best_matrix_error.append(current_row)
#     good_matrix_error.append(current_row2)
#     bad_matrix_error.append(current_row3)
# beta=stats.t(df=nb_experiments-1).ppf(0.95)/np.sqrt(nb_experiments-1)
# mean_error=np.mean(best_matrix_error,axis=0)
# std_error=np.std(best_matrix_error,axis=0)
# mean_error2=np.mean(good_matrix_error,axis=0)
# std_error2=np.std(good_matrix_error,axis=0)
# mean_error3=np.mean(bad_matrix_error,axis=0)
# std_error3=np.std(bad_matrix_error,axis=0)
# plt.figure()
# plt.xscale("log")
# plt.yscale("log")
# plt.plot(size_range,mean_error,marker='|', label=r'$\bar{M}$',linewidth=3,markersize=12)
# plt.fill_between(size_range, (mean_error-beta*std_error), (mean_error+beta*std_error), color='b', alpha=.1)
# plt.plot(size_range,mean_error2, marker='.', label=r'$\hat{M}$',linewidth=3,markersize=12)
# plt.fill_between(size_range, (mean_error2-beta*std_error2), (mean_error2+beta*std_error2), color='orange', alpha=.1)
# plt.plot(size_range,mean_error3, marker='_', label=r'$\tilde{M}$',linewidth=3,markersize=12)
# plt.fill_between(size_range, (mean_error3-beta*std_error3), (mean_error3+beta*std_error3), color='green', alpha=.1)
# plt.xlabel("Matrix Size",fontsize=30)
# plt.xlim(2,296) 
# plt.ylabel("Max-Norm Error",fontsize=30) 
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.legend(loc='best',fontsize=25)
# plt.show()

#%%
#Experiment 6: Comparison between RS-BPI,SBPI,Benchmark. Uncomment to run.

# nb_experiments=50
# pr=1-1/n
# x_range=(np.arange(10,sqrt(T),3)**2).astype(int)
# best_value=best_policy_value(pr,true_matrix)
# policy_value1=[]
# policy_value2=[]
# policy_value3=[]

# for i in range(nb_experiments):
#     samples=samples_generation(T,pr,true_matrix)
#     policy_value1.append([RS_MLB_BPI(samples,x,pr,true_matrix) for x in x_range])
#     policy_value2.append([RS_BPI(samples,x,pr,true_matrix) for x in x_range]) 
#     policy_value3.append([IPS_BPI(samples,x,pr,true_matrix) for x in x_range])
# beta=stats.t(df=nb_experiments-1).ppf(0.95)/np.sqrt(nb_experiments-1)
# mean_value=np.mean(policy_value1,axis=0)
# std_value=np.std(policy_value1,axis=0)
# mean_value2=np.mean(policy_value2,axis=0)
# std_value2=np.std(policy_value2,axis=0)
# mean_value3=np.mean(policy_value3,axis=0)
# std_value3=np.std(policy_value3,axis=0)


# plt.figure()

# plt.yscale("linear")
# plt.plot(np.sqrt(x_range),mean_value,marker='|',label='RS-BPI',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_value-beta*std_value), (mean_value+beta*std_value), color='b', alpha=.1)
# plt.plot(np.sqrt(x_range),mean_value2, marker='.', label='SBPI',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_value2-beta*std_value2), (mean_value2+beta*std_value2), color='orange', alpha=.1)
# plt.plot(np.sqrt(x_range),mean_value3, marker='_', label='Benchmark',linewidth=3,markersize=12)
# plt.fill_between(np.sqrt(x_range), (mean_value3-beta*std_value3), (mean_value3+beta*std_value3), color='green', alpha=.1)
# plt.plot(np.sqrt(x_range),[best_value for x in x_range],label='Maximal Policy Value',linewidth=3)
# plt.xlabel(r'$\sqrt{T}$',fontsize=30)
# plt.ylabel("Value of Learned Policy",fontsize=30) 
# plt.xlim(4,140)
# plt.xticks(fontsize=25)
# plt.yticks(fontsize=25)
# plt.legend(loc='best',fontsize=25)
# plt.show()
