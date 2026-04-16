def main():

    # -------------------------------
    # MODE INPUT (loop until correct)
    # -------------------------------
    while True:
        print("\n===== SELECT MODE =====")
        mode_choice = input("Mode (scratch/sklearn): ").lower().strip()

        if mode_choice in ["scratch", "sklearn"]:
            break
        else:
            print("❌ Invalid mode. Try again.")

    # -------------------------------
    # TASK INPUT (loop until correct)
    # -------------------------------
    while True:
        print("\n===== SELECT TASK =====")
        print("1. Regression")
        print("2. Classification")

        task_choice = input("Enter choice (1/2): ").strip()

        if task_choice in ["1", "2"]:
            break
        else:
            print("❌ Invalid task. Try again.")

    # -------------------------------
    # EXECUTION
    # -------------------------------
    if mode_choice == "scratch":

        if task_choice == "1":
            print("\n===== SCRATCH REGRESSION =====")
            from scratch.linear_regression_numpy import run_regression_scratch
            run_regression_scratch()

        elif task_choice == "2":
            print("\n===== SCRATCH CLASSIFICATION =====")
            from scratch.logistic_regression_numpy import run_titanic_scratch
            run_titanic_scratch()

    elif mode_choice == "sklearn":

        if task_choice == "1":
            print("\n===== SKLEARN REGRESSION =====")
            from sklearn_pipeline.house_prices import run_regression
            run_regression()

        elif task_choice == "2":
            print("\n===== SKLEARN CLASSIFICATION =====")
            from sklearn_pipeline.titanic_pipeline import run_titanic
            run_titanic()


if __name__ == "__main__":
    main()