import pytest
import juml

def test_load_model():
    command = get_command()
    command.run(**command.get_kwargs())
    model = command.init_object("model")
    dataset = command.init_object("dataset")
    assert isinstance(model, juml.models.FeedForwardModel)
    assert isinstance(dataset, juml.data.ClassificationDataset)

    x_test, t_test = dataset.get_full_batch("test")
    x_test, t_test = dataset.format_batch(x_test, t_test)
    y_test = model.forward(x_test)
    test_acc = juml.util.multiclass_acc(y_test, t_test).item()
    assert test_acc == 1.0

    command = get_command()
    model = command.init_object("model", input_dim=2, output_dim=2)
    assert isinstance(model, juml.models.FeedForwardModel)

    y_test = model.forward(x_test)
    test_acc = juml.util.multiclass_acc(y_test, t_test).item()
    assert test_acc < 1.0

    command = get_command()
    with pytest.raises(TypeError):
        loaded_model = command.load_model()

    loaded_model = command.load_model(input_dim=2, output_dim=2)
    assert isinstance(loaded_model, juml.models.FeedForwardModel)

    y_test = loaded_model.forward(x_test)
    test_acc = juml.util.multiclass_acc(y_test, t_test).item()
    assert test_acc == 1.0

def test_load_table_data():
    command = get_command()
    command.run(**command.get_kwargs())

    new_command = get_command()
    test_acc = new_command.load_table_data("test_acc")

    assert isinstance(test_acc, list)
    assert len(test_acc) == 200
    assert test_acc[0] < 1.0
    assert test_acc[-1] == 1.0

def get_command() -> juml.commands.Command:
    command = juml.commands.TrainClassification.init_framework_command()
    command.update(
        {
            "seed": 0,
            "epochs": 200,
            "batch_size": 100,
            "save_model": True,
            "dataset": "Xor",
            "model": "ReluMlp",
            "model.ReluMlp.hidden_dim": 10,
            "model.ReluMlp.depth": 2,
        },
    )
    return command
