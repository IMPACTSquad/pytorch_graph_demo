import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric.transforms as T

from Graph import AirQualityClassification, AirQualityRegression

activations = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "softmax": F.softmax,
}


def subjective_output_activation(x):
    return torch.exp(x) + 1


output_activations = {
    "softmax": torch.nn.Softmax(dim=1),
    "sigmoid": torch.nn.Sigmoid(),
    "linear": None,
    "subjective": subjective_output_activation,
}


class Subjective_GCN(torch.nn.Module):
    def __init__(
        self,
        n_labels,
        channels=16,
        dropout_rate=0.5,
        activation="relu",
        output_activation="subjective",
        use_bias=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout_rate = dropout_rate
        self._gcn0 = GCNConv(
            in_channels=-1,  # -1 means that it is inferred from the data automatically
            out_channels=channels,  # number of output channels of layer
            cached=True,  # stores \(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}\) for faster computation NOTE only works if the graph is static, i.e. transductive learning scenario
            bias=use_bias,  # whether to use a bias vector
        )
        self._gcn1 = GCNConv(
            in_channels=-1,  # -1 means that it is inferred from the data automatically
            out_channels=n_labels,  # number of model output features
            cached=True,
            bias=use_bias,
        )
        self._activation = activations[activation]
        self._output_activation = output_activations[output_activation]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # dropout
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # first gcn layer
        x = self._gcn0(x, edge_index)
        # first layer's non-linear activation function
        x = self._activation(x)

        # dropout
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # second gcn layer
        x = self._gcn1(x, edge_index)

        if self._output_activation is not None:
            x = self._output_activation(x)
        return x


class SquaredErrorDirichlet(torch.nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
    ):
        self.reduction = reduction

    def forward(self, input, target):
        return subjective_loss(input, target)


def subjective_loss(input, target):
    oh_target = F.one_hot(target, input.shape[1]).float()
    strength = torch.sum(input, dim=1, keepdim=True)
    prob = input / strength
    loss = torch.square(prob - oh_target) + prob * (1 - prob) / (strength + 1.0)
    loss = torch.sum(loss, dim=1)  # sum over classes
    return loss.mean()  # average over nodes


def train_classification():
    learning_rate = 1e-2
    seed = 1
    epochs = 100
    patience = 50
    n_classes = 15
    print_every = 10  # print loss every x epochs

    torch.manual_seed(seed)  # make weight initialization reproducible

    # use cpu/gpu depending on availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = AirQualityClassification(
        seed=seed, pre_transform=T.KNNGraph(k=10, force_undirected=True), train_ratio=0.2, val_ratio=0.1
    )

    # GCN model
    model = Subjective_GCN(n_labels=n_classes, channels=32, output_activation="subjective", use_bias=True)
    # Put model/data on CPU/GPU as needed
    model = model.to(device)
    data = dataset[0].to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2.5e-4)
    # loss function
    criterion = subjective_loss  # subjective logic loss (for classification)

    # Put model in training mode (determines whether dropout is used etc.)
    model.train()

    best_val_loss = None
    val_loss_patience = 0

    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear gradients.

        output = model(data)  # Perform a single forward pass.

        loss = criterion(
            output[dataset.mask_tr], data.y[dataset.mask_tr]
        )  # Compute the loss solely based on the training nodes.

        loss.backward()  # Derive gradients.

        optimizer.step()  # Update parameters based on gradients (back-propagation).

        with torch.no_grad():  # don't track gradients during validation
            output = model(data)
            val_loss = criterion(
                output[dataset.mask_va], data.y[dataset.mask_va]
            )  # Compute the loss solely based on the validation nodes.
            if (
                best_val_loss is None or val_loss <= best_val_loss
            ):  # if validation loss is better than previous best, reset patience
                torch.save(model, "models/best_subjective_classifier.pt")  # checkpoint model
                best_val_loss = val_loss
                val_loss_patience = 0
            else:
                val_loss_patience += 1
                if (
                    val_loss_patience >= patience
                ):  # if validation loss hasn't improved in patience epochs, stop training
                    break

        if (epoch + 1) % print_every == 0:  # every print_every epochs
            print(f"Epoch {epoch + 1}: loss: {loss:.3f} val_loss: {val_loss:.3f}")

    # restore best model
    model = torch.load("models/best_subjective_classifier.pt")
    # Put model in evaluation mode (determines whether dropout is used etc. i.e. no dropout in evaluation mode)
    model.eval()

    # Evaluate model
    print("Evaluating model.")
    predictions = model(data).cpu().detach().numpy()
    test_acc = np.mean(predictions.argmax(axis=1)[dataset.mask_te] == dataset[0].y[dataset.mask_te].numpy())
    print("Done.\n" + f"Test Accuracy: {100 * test_acc:.2f}%")
    return model, data, (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
