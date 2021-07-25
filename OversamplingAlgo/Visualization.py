# Necessary packages
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
# from visualization_metrics import visualization
   
def visualization (ori_data, generated_data, analysis):
  # Analysis sample size (for faster computation)
  anal_sample_no = min([1000, len(ori_data)])
  idx = np.random.permutation(len(ori_data))[:anal_sample_no]
  # print(ori_data)
  # Data preprocessing
  ori_data = np.asarray(ori_data)
  generated_data = np.asarray(generated_data)  

  ori_data = ori_data[idx]
  generated_data = generated_data[idx]
  print(ori_data.shape)
  no, dim = ori_data.shape  
        
  colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)]    

  if analysis == 'pca':
    # PCA Analysis
    pca = PCA(n_components = 2)
    pca.fit(ori_data)
    pca_results = pca.transform(ori_data)
    pca_hat_results = pca.fit_transform(generated_data)
    # visualization(ori_data, generated_data, 'pca')
    # Plotting
    f, ax = plt.subplots(1)    
    plt.scatter(pca_results[:,0], pca_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

    ax.legend()  
    plt.title('PCA plot')
    plt.xlabel('x-pca')
    plt.ylabel('y_pca')
    plt.show()
    
  elif analysis == 'tsne':
    
    # Do t-SNE Analysis together       
    prep_data_final = np.concatenate((ori_data, generated_data), axis = 0)

    # TSNE anlaysis
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 40, n_iter = 300)
    tsne_results = tsne.fit_transform(ori_data)
    tsne_results1=tsne.fit_transform(generated_data)
    # colors = ["red" for i in range(anal_sample_no)] + ["blue" for i in range(anal_sample_no)] 
    # Plotting
    f, ax = plt.subplots(1)
      
    plt.scatter(tsne_results[:,0], tsne_results[:,1],
                c = colors[:anal_sample_no], alpha = 0.2, label = "Original")
    plt.scatter(tsne_results1[:,0], tsne_results1[:,1], 
                c = colors[anal_sample_no:], alpha = 0.2, label = "Synthetic")

    ax.legend()
      
    plt.title('t-SNE plot')
    plt.xlabel('x-tsne')
    plt.ylabel('y_tsne')
    plt.show()