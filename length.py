import numpy as np
from sklearn.cluster import KMeans

def two_blob_kmeans(bw_images_array):
    xs = []
    ys = []

    for frame in bw_images_array:
        # Get coordinates of all white pixels
        ys_white, xs_white = np.where(frame == 255)

        # If not enough white pixels → blank frame
        if len(xs_white) < 2:
            xs.append((np.nan, np.nan))
            ys.append((np.nan, np.nan))
            continue

        coords = np.column_stack((xs_white, ys_white))  # shape: (N, 2)

        # If too few points for 2 clusters, KMeans fails → skip
        if coords.shape[0] < 2:
            xs.append((np.nan, np.nan))
            ys.append((np.nan, np.nan))
            continue

        # Run k-means to split into two clusters
        kmeans = KMeans(n_clusters=2, n_init=10)
        labels = kmeans.fit_predict(coords)

        # Cluster centroids
        c = kmeans.cluster_centers_

        # c is shape (2, 2): [ [x1,y1], [x2,y2] ]
        xs.append((c[0][0], c[1][0]))
        ys.append((c[0][1], c[1][1]))

    return xs, ys