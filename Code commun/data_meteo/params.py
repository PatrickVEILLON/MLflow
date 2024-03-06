""" params module """

param_grid_lr   = { "C"                 : [ 10**(i) for i in range(-4, 5) ],
                    "solver"            : [ "lbfgs", "liblinear", "newton-cg", "sag", "saga" ] }

param_grid_svm  = { "C"                 : [ 10**(i) for i in range(-4, 5) ],
                    "gamma"             : [0.001, 0.01, 0.1, 1],
                    "kernel"            : [ "linear", "poly", "rbf" ] }

param_grid_knn  = { "n_neighbors"       : [list(range(1, 32))],
                    "metric"            : [ "euclidean", "manhattan", "minkowski" ] }

param_grid_tree = { "criterion"         : [ "entropy", "gini" ],
                    "max_depth"         : [ list(range(1, 10)) ],
                    "max_features"      : [ "sqrt", "log2" ] }

param_grid_rf   = { "n_estimators"      : [10, 50, 100, 250, 500, 1000],
                    "min_samples_leaf"  : [1, 3, 5, 7], 
                    "max_features"      : ["sqrt", "log2"]}
