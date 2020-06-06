import numpy as np
from PIL import Image as pil_image
from sklearn.cluster import MiniBatchKMeans, Birch


def load_img(bytes_io):
    return pil_image.open(bytes_io).convert('RGB')


class ColorExtractor:
    def __init__(
        self,
        algo='kmeans',
        n_colors=4,
        random_state=0,
        batch_size=100,
        init='k-means++',
        max_iter=100,
        verbose=0,
        compute_labels=True,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=3,
        reassignment_ratio=0.01,
        threshold=0.5,
        branching_factor=50,
        copy=True
    ):
        if algo == 'birch':
            self.extractor = BirchColorExtractor(
                n_colors=n_colors,
                threshold=threshold,
                branching_factor=branching_factor,
                compute_labels=compute_labels,
                copy=copy
            )
        else:
            self.extractor = KmeansColorExtractor(
                n_colors=n_colors,
                random_state=random_state,
                batch_size=batch_size,
                init=init,
                max_iter=max_iter,
                verbose=verbose,
                compute_labels=compute_labels,
                tol=tol,
                max_no_improvement=max_no_improvement,
                init_size=init_size,
                n_init=n_init,
                reassignment_ratio=reassignment_ratio
            )

    def extract(self, img):
        return self.extractor.extract(img)


class KmeansColorExtractor:
    def __init__(
        self,
        n_colors=4,
        random_state=0,
        batch_size=100,
        init='k-means++',
        max_iter=100,
        verbose=0,
        compute_labels=True,
        tol=0.0,
        max_no_improvement=10,
        init_size=None,
        n_init=3,
        reassignment_ratio=0.01
    ):
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_colors,
            random_state=random_state,
            batch_size=batch_size,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            compute_labels=compute_labels,
            tol=tol,
            max_no_improvement=max_no_improvement,
            init_size=init_size,
            n_init=n_init,
            reassignment_ratio=reassignment_ratio
        )

    def extract(self, img):
        img_array = np.array(img, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        w, h, d = tuple(img_array.shape)
        assert d == 3
        image_array = np.reshape(img_array, (w * h, d))

        print("Fitting model on a small sub-sample of the data")
        # manually fit on batches
        self.kmeans.fit(image_array)

        # Get labels for all points
        print("Predicting color indices on the full image (k-means)")
        labels = self.kmeans.labels_

        main_color_array = 255 * self.kmeans.cluster_centers_
        return [
            dict(
                color=dict(
                    r=color[0],
                    g=color[1],
                    b=color[2]
                ),
                count=labels[labels == i].shape[0]
            ) for i, color in enumerate(main_color_array)
        ]


class BirchColorExtractor:
    def __init__(
        self,
        n_colors=None,
        threshold=0.5,
        branching_factor=50,
        compute_labels=True,
        copy=True
    ):
        self.birch = Birch(
            n_clusters=n_colors,
            threshold=threshold,
            branching_factor=branching_factor,
            compute_labels=compute_labels,
            copy=copy
        )

    def extract(self, img):
        img_array = np.array(img, dtype=np.float64) / 255

        # Load Image and transform to a 2D numpy array.
        w, h, d = tuple(img_array.shape)
        assert d == 3
        image_array = np.reshape(img_array, (w * h, d))

        print("Fitting model on a small sub-sample of the data")
        # manually fit on batches
        self.birch.fit(image_array)

        # Get labels for all points
        print("Predicting color indices on the full image (birch)")
        labels = self.birch.labels_

        main_color_array = 255 * self.birch.subcluster_centers_
        return [
            dict(
                color=dict(
                    r=color[0],
                    g=color[1],
                    b=color[2]
                ),
                count=labels[labels == i].shape[0]
            ) for i, color in enumerate(main_color_array)
        ]
