# from scratch.logistic_regression_numpy import run_titanic_scratch
def main():
    mode = "sklearn"   # change here
    # mode = "scratch"   # change here

    if mode == "scratch":
        print("\n===== TITANIC =====")
        from scratch.logistic_regression_numpy import run_titanic_scratch
        run_titanic_scratch()

    elif mode == "sklearn":
        print("\n===== TITANIC =====")
        from sklearn_pipeline.titanic_pipeline import run_titanic
        run_titanic()

        # print("\n===== IRIS =====")
        # run_iris()
        
        # print("\n===== REGRESSION =====")
        # run_regression()
        
        # print("\n===== CLUSTERING =====")
        # run_clustering()

if __name__ == "__main__":
    main()