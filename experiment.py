from helper.helper import Helper


if __name__ == "__main__":

    helper = Helper(vae_only=True,
                    hidden_values=[2, 4, 8, 16, 32, 64],
                    reg_values=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                    drp_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    epochs=300)

    helper.train_models()

    helper.make_tests()

    helper.get_best_reconstruction()
    helper.show_reconstrunction(vae=True, hidden=32, reg_val=1e-05, drp_val=1e-01, imgs=5)

    helper.get_best_precision()
    helper.get_best_recall()
    helper.get_best_f1()

    print(f"Best Model Threshold is: {helper.get_th(vae=True, hidden=32, reg_val=1e-5, drp_val=0.5)}")

    helper.get_best_svmoc()


