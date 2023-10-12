import time
import pandas as pd
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import KNeighborsClassifier as ConcreteKNeighborsClassifier

### Data-set generation
def generate_dataset():
    def make_dataset(type, n, n_features):
        print(n, n_features)
        if type == "moon":
            X, y = make_moons(n_samples=(n, n_features), noise=0.2)
        elif type == "circles":
            X, y = make_circles(n_samples=(n, n_features), noise=0.2)
        elif type == "classif":
            X, y = make_classification(n_samples=n, n_features=n_features, n_classes=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        return X_train, X_test, y_train, y_test

    dataset_size = [50]
    dims = [20]
    datasets = []

    for n in dataset_size:
        for dim in dims:
            # Split the data-set into a train and testing sets

            # X_train, X_test, y_train, y_test = make_dataset(type="moon", n=n, n_features=dim)
            # datasets.append([X_train, X_test, y_train, y_test, "moon"])

            # X_train, X_test, y_train, y_test = make_dataset(type="circles", n=n, n_features=dim)
            # datasets.append([X_train, X_test, y_train, y_test, "circle"])
            X_train, X_test, y_train, y_test = make_dataset(type="classif", n=n, n_features=dim)
            datasets.append([X_train, X_test, y_train, y_test, "classif"])

    return datasets


def read_history():
    return pd.read_csv("history.txt")


# Stat testing
if __name__ == "__main__":

    datasets = generate_dataset()
    n_neighbors = 3
    list_bits = [3]
    list_rounding = [4, 5, -1, 6]

    list_fhe_exec_time = []
    list_compilation_time = []
    list_key_gey_time = []
    list_score_sk, list_score_simulate, list_score_fhe = [], [], []
    list_bitwidth_circuit = []

    verbose = True
    # with open("history.txt", "w") as f:
    #     f.write("type,n,dims,qbit,rounding,cbit,compile_time,key_time,sk_score, simul_score,fhe_time\n")
                        
    for dataset in datasets:
        X_train, X_test, y_train, y_test, t = dataset
    
        for bit in list_bits:
            for rounding in list_rounding:
                if verbose:
                    print(f"{t} - {X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}, {rounding=}, {bit=}")
                
                concrete_knn = ConcreteKNeighborsClassifier(
                    n_bits=bit, n_neighbors=n_neighbors, rounding_threshold_bits=rounding
                )
                # Fit both the Concrete ML and its equivalent float estimator on clear data
                concrete_knn, sklearn_model = concrete_knn.fit_benchmark(X_train, y_train)
                # Compilation
                time_begin = time.time()
                circuit = concrete_knn.compile(X_train)
                end_time = time.time() - time_begin
                cbit = circuit.graph.maximum_integer_bit_width()
                if verbose:
                    print(f"Compilation time: {end_time:.2f} seconds - "
                        f"maximum_bit_width: {cbit}")

                list_compilation_time.append(end_time)
                list_bitwidth_circuit.append(cbit)

                with open("history.txt", "a") as f:
                    f.write(f"{t},{X_train.shape[0]},{X_train.shape[1]},{bit},{rounding},{cbit},"
                    f"{list_compilation_time[-1]},")

                # Key generation
                time_begin = time.time()
                circuit.client.keygen()
                end_time = time.time() - time_begin
                if verbose:
                    print(f"Key generation time: {end_time:.2f} seconds")
                list_key_gey_time.append(end_time)

                with open("history.txt", "a") as f:
                    f.write(f"{list_key_gey_time[-1]},")

                # scikit-learn inference
                predict_sklearn = sklearn_model.predict(X_test)
                score_sklearn = accuracy_score(y_test, predict_sklearn)

                # b- FHE simulation inference
                pred_cml_simulate = concrete_knn.predict(X_test, fhe="simulate")
                score_cml_simulate = accuracy_score(y_test, pred_cml_simulate)

                list_score_sk.append(score_sklearn)
                list_score_simulate.append(score_cml_simulate)

                with open("history.txt", "a") as f:
                    f.write(f"{list_score_sk[-1]},{list_score_simulate[-1]},")
                    
                # c- FHE inference
                time_begin = time.time()
                pred_cml_fhe = concrete_knn.predict(X_test[0, None], fhe="execute")
                end_time = (time.time() - time_begin)
                if verbose:
                    print(f"FHE inference execution time: {end_time:.2f}s per sample")
                
                list_fhe_exec_time.append(end_time)

                if verbose:
                    print(f"{list_score_sk[-1]=} - {list_score_simulate[-1]=} - Exec time = {list_fhe_exec_time[-1]=}\n")

                with open("history.txt", "a") as f:
                    f.write(f"{list_fhe_exec_time[-1]}\n")
                                    