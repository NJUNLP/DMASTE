import torch 

class FeedForward(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, dropout):
        super(FeedForward, self).__init__()
        hidden_dims = [hidden_dim] * num_layers  # type: ignore
        activations = [activation] * num_layers  # type: ignore
        dropout = [dropout] * num_layers  # type: ignore
        self._activations = torch.nn.ModuleList(activations)
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            a = torch.nn.Linear(layer_input_dim, layer_output_dim)
            torch.nn.init.xavier_normal_(a.weight)
            linear_layers.append(a)
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        output = inputs
        for layer, activation, dropout in zip(
            self._linear_layers, self._activations, self._dropout
        ):
            output = dropout(activation(layer(output)))
        return output