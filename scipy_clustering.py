1. Heirarichal clustering using scipy

from scipy.cluster.hierarchy import linkage, fcluster
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method = 'ward', metric = 'euclidean') # Use the linkage() function
comic_con['cluster_labels'] = fcluster(distance_matrix, 2, criterion='maxclust') # Assign cluster labels
sns.scatterplot(x='x_scaled', y='y_scaled', hue='cluster_labels', data = comic_con) # Plot clusters
plt.show()


2. K means using scipy

from scipy.cluster.vq import kmeans, vq
centroids,_ = kmeans(df, 2) # Compute cluster centers
df['cluster_labels'], _ = vq(df, centroids) # Assign cluster labels
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df) # Plot the points with seaborn
plt.show()

3. In clustering it is recommended to scale the data before performing clustering otheriwise there are chances of having a bias
introduced in your model due to dominance of a variable. Scaling helps cut down large variations and thus we get features that have 
std =1

You can do this using whiten method of scipy

from scipy.cluster.vq import whiten
goals_for = [4,3,2,3,1,1,2,0,1,4]
scaled_data = whiten(goals_for) # it divided each column by its std in this case we just have a single column

Ex 2 for a dataframe
fifa['scaled_wage'] = whiten(fifa['eur_wage'])
fifa['scaled_value'] = whiten(fifa['eur_value'])

4. The linkage method takes in observation as the argument along with metric and method . The metric is the way using which the 
distance is going to be calculated and it could be as simple as "euclidean" . The method determines how to combine different 
point/clusters , they are of different types:
    - "single" based on 2 closet objects
    - "complete" based on 2 farthest objects
    - "average" based on airthmatic mean
    - "centroid" based on gometric mean
    - "median" based on median on objects
    - "ward" based on sum of squares, focuses upon clusters more contentric towards its center
    
5. fcluster takes in distance matrix generated using linkage method , num of clusters needed and criterion to decide threhold to
form clusters.


6. to decide upon the no of clusters you can create dendograms

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage, fcluster
distance_matrix = linkage(comic_con[['x_scaled', 'y_scaled']], method = 'ward', metric = 'euclidean') 
dn = dendrogram(distance_matrix)
plt.show()
   
    
7. Heririchal clustering is has a lot of time complexity which increases in a quadratic manner as the no of observation increases

8. In kmeans clustering you can find the best value for the no of clusters using the elbow plot which si constructed by
measureing the intertia/distortions as a function of the no of clusters. Inertia/distortions can be defined as 
sum of squares of distnaces of points from its cluster centers, this value dec with inc no of clusters and becomes 0 when
no of clusters equal no of observations. If the data is uniformly distributed then elbow plot might not help and it gives wierd
plots

Other menthods to do that is using Gap statistics and average sihouette


9. Challenges with kmeans:
    - Prior knowledge of k
    - Impact of seeds, initilisation effects the way clusters are generated
    - Formation of equal sized clusters to reduce distortions
    
    
    
