# First create the testdata
using Distributions
function simDat(nObs,nLoci,bMean,bStd,resStd)
X=[ones(nObs,1) sample([0,1,2],(nObs,nLoci))]
b=rand(Normal(bMean,bStd),size(X,2))
y=X*b+rand(Normal(0.0,resStd),nObs)
return(y,X,b)
end

nObs =100;
nLoci =5;
bMean =0.0;
bStd =0.5;
resStd =1.0;
res=simDat(nObs,nLoci,bMean,bStd,resStd);
y=res[1];
X=res[2];
b=res[3];
# Now do the analysis using MCMC

# Provide all the parameters with starting values
niter=100000;                       # define the length of the Markov chain
numParms=size(X,2);                # Find out how many parameters need sampling
allSamples=zeros(niter,numParms);  # Reserve space for storing the samples
thisSample=zeros(numParms);        # Set initial samples to plausible values
diagXpX=diag(X'X);                 # Form the elements need for the var(b-hat)
sigmaSqE=resStd^2;                 # In this example we know the residual variance
burnIn=10;                         # Optionally discard the first few samples

# Remember the model equation    y = Xb + e
work = y - X*thisSample;  #  Create a work vector of phenotypes adjusted for everything

for (iter in 1:niter)
   for (thisParm in 1:numParms)
       work = work + X[:,thisParm]*thisSample[thisParm]       # unadjust for the current sample
       leastSq = dot(X[:,thisParm],work)/diagXpX[thisParm]    # get OLS value
       thisSample[thisParm]=rand(Normal(leastSq, sqrt(sigmaSqE/diagXpX[thisParm])))
       work = work - X[:,thisParm]*thisSample[thisParm]       # adjust for the new sample
    end  # loop over all the parameters
    allSamples[iter,:]= thisSample                            # Store the current samples
    if iter%100 == 0 
        work = y - X*thisSample                               # Avoid rounding errors
    end
end # loop over the number of iterations

# Calculate the OLS solutions for comparison
ols=X'X\X'y;
XpXinv = inv(X'X);

function results()
  println("      MCMC               OLS")
  println("   b-hat  var(b-hat)   b-hat var(b-hat)")
  for (thisParm in 1:numParms)
      betaHat=mean(allSamples[burnIn:end,thisParm])     # Compute posterior mean 
      varBetaHat=var(allSamples[burnIn:end,thisParm])   # Compute posterior variance
      @printf " %8.4f %8.4f %8.4f %8.4f \n" betaHat varBetaHat ols[thisParm] diag(XpXinv)[thisParm]*sigmaSqE
  end
end

results()  # Now print out the results
    