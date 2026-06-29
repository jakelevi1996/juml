import juml

def test_mlp_xor():
    command = juml.commands.TrainClassification.init_framework_command()
    command.update(
        {
            "seed": 0,
            "epochs": 1,
            "batch_size": 100,
            "dataset": "Xor",
            "model": "ReluMlp",
            "model.ReluMlp.hidden_dim": 10,
            "model.ReluMlp.depth": 2,
        },
    )
    command.run(**command.get_kwargs())
    assert command.load_metric("final_test_acc") <= 0.5

    command.update({"epochs": 200})
    command.run(**command.get_kwargs())
    assert command.load_metric("final_test_acc") == 1.0
