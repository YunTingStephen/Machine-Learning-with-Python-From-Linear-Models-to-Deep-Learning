import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K = [1,2,3,4]  #Clusters to try
seeds = [0,1,2,3,4] #Seeds to try

#Costs for different seeds
cost_kmeans = [0,0,0,0,0]
cost_EM = [0,0,0,0,0]

#Best Seeds for Algo
bestseed_kmeans = [0,0,0,0]
bestseed_EM = [0,0,0,0]

#Mixture for Best Seed for Algo 
mixture_kmeans = [0,0,0,0,0]
mixture_EM = [0,0,0,0,0]

# Posterior probs. for best seeds
post_kmeans = [0, 0, 0, 0, 0]
post_EM = [0,0,0,0,0]

# BIC score of cluster
bic = [0., 0., 0., 0.]

for k in range(len(K)):
    for i in range(len(seeds)):
        mixture_kmeans[i], post_kmeans[i], cost_kmeans[i] = kmeans.run(X, *common.init(X,K[k], seeds[i]))
        mixture_EM[i], post_EM[i], cost_EM[i] = naive_em.run(X, *common.init(X, K[k], seeds[i]))



    print("=============== Clusters:", k+1, "======================")
    print("Lowest cost using kMeans is:", np.min(cost_kmeans))
    print("Lowest cost using EM is:", np.max(cost_EM))

    #Save best seed for plotting
    bestseed_kmeans[k] = np.argmin(cost_kmeans)
    bestseed_EM[k] = np.argmax(cost_EM)
    
    common.plot(X, mixture_kmeans[bestseed_kmeans[k]],
                post_kmeans[bestseed_kmeans[k]],
                title="kmeans")

    common.plot(X, mixture_EM[bestseed_EM[k]],
                post_EM[bestseed_EM[k]],
                title="EM")
    bic[k] = common.bic(X, mixture_EM[bestseed_EM[k]], np.max(cost_EM))


# Print the best K based on BIC
print("================= BIC ====================")
print("Best K is:", np.argmax(bic)+1)
print("BIC for the best K is:", np.max(bic))
 
########## End: kMeans vs EM (and BIC) #############


