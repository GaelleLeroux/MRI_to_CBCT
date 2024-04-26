import neptune

run = neptune.init_run()

run["algorithm"] = "ConvNet"

params = {
    "activation": "sigmoid",
    "dropout": 0.20,
    "learning_rate": 0.1,
    "n_epochs": 100,
}
run["model/parameters"] = params

run["model/parameters/activation"] = "ReLU"
run["model/parameters/batch_size"] = 64

for epoch in range(params["n_epochs"]):
    # this would normally be your training loop
    run["train/loss"].append(0.99**epoch)
    run["train/acc"].append(1.01**epoch)
    run["eval/loss"].append(0.98**epoch)
    run["eval/acc"].append(1.02**epoch)

run["data_versions/train"].track_files("sample.csv")
run["data_sample"].upload("sample.csv")

run["f1_score"] = 0.95
run.stop()