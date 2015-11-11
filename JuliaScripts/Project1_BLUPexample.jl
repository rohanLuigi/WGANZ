# Mixed Model Evaluation
sigmaSqE=3;
sigmaSqG=1;
sigmaSqP=sigmaSqG+sigmaSqE;
heritability=sigmaSqG/sigmaSqP;
lambda=(1-heritability)/heritability;  # or lambda=sigmaSqE/sigmaSqG

# clone the Pedigree Module - only need to do this once
#Pkg.clone("https://github.com/reworkhow/PedModule.jl.git")
using PedModule
using Distributions

pedigree = [1 0 0
            2 0 0
            3 0 0
            4 1 2 
            5 1 2 
            6 1 3]

writedlm("pedFile",pedigree)

ped=PedModule.mkPed("pedFile")
ped.idMap
PedModule.getIDs(ped)         # Access the reordered IDs
Ainv=PedModule.AInverse(ped)  # Note this is a sparse matrix








# full() converts a sparse matrix to a dense matrix
# round() reduces the number of decimal places
A=round(inv(full(Ainv)),2)

numAnimals=size(Ainv,1)

X=ones(numAnimals,1);
Z=eye(numAnimals);
numFixed=size(X,2)

















srand(2)      # Set the integer seed for the random number generator
b=rand(Normal(0,sqrt(sigmaSqP)),numFixed);
d=MvNormal(zeros(numAnimals),A*sigmaSqG)
u=rand(d,1)
e=rand(Normal(0.0,sqrt(sigmaSqE)),numAnimals);

y=X*b + Z*u + e














# May want to make some randomly missing (delete rows of y, X, Z and e)








mmeLhs=[X'X X'Z
        Z'X (Z'Z+Ainv*lambda)]
mmeRhs=[X'y
        Z'y]
 










             
# Now choose among various solvers
# Here we will use direct solution to get PEV
# Alternatives are PCG (iterative or indirect solution)
# MCMC (just like the code we already developed)        
                
mmeInv=inv(mmeLhs)
soln=mmeInv*mmeRhs
pev=(diag(mmeInv)*sigmaSqE)[(numFixed+1):end]
reliability=1-pev/sigmaSqG
accuracy=sqrt(reliability)


# Note this may be different to sqrt(heritability) due to 
# i) chance sampling that creates covariance between u and e
# ii) loss of information from the fixed effects 

# GLS solution

G=A*sigmaSqG
R=eye(numAnimals)*sigmaSqE
V=Z*G*Z' + R
Vinv=inv(V)
GLS_LHS=X'Vinv*X
GLS_RHS=X'Vinv*y
GLS_bhat=GLS_LHS\GLS_RHS

y_adj=y-X*GLS_bhat

# BLUP via GLS
GLS_uhat=G*Z'Vinv*(y_adj)

# Linear functions (ie contrasts) of BLUP
K=[1 -.5 -.5 0 0 0    
   0   1  -1 0 0 0]  # Define the linear contrast
   
# BLUP of the linear contrast
K*soln[numFixed+1:end]
   
# Variance of the linear constrasts
varKu=K*G*K'

# Variance of the BLUP of linear contrasts
varKuhat=K*(G-mmeInv[numFixed+1:end,numFixed+1:end])*K'

# Reliability of the linear contrasts
relKuhat=diag(varKuhat)./diag(varKu)
 