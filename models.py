import torch.nn as nn

class MLP_Arquitectura1(nn.Module):
    """
    Arquitectura 1: 1 capa oculta con 16 neuronas y función de activación Sigmoid.
    Modelo básico, baja capacidad, menor riesgo de overfitting.
    """
    def __init__(self, input_dim = 10, output_dim = 2):
        super().__init__()
        self.hidden = nn.Linear(input_dim, 16)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(16, output_dim)

    def forward(self, x):
        x = self.activation(self.hidden(x))
        return self.output(x)


class MLP_Arquitectura2(nn.Module):
    """
    Arquitectura 2: 2 capas ocultas con [32, 16] neuronas por cada capa y con función de activación ReLU.
    Mayor profundidad. ReLU evita el desvanecimiento del gradiente que sufre Sigmoid en capas internas.
    """
    def __init__(self, input_dim = 10, output_dim = 2):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, 32)
        self.hidden2 = nn.Linear(32, 16)
        self.activation = nn.ReLU()
        self.output = nn.Linear(16, output_dim)

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        return self.output(x)


class MLP_Arquitectura3(nn.Module):
    """
    Arquitectura 3: 2 capas ocultas con [64, 32] neuronas por cada capa con ReLU y Dropout(0.3).
    Mayor capacidad que la Arquitectura 2, compensada con Dropout para reducir el sobreajuste en un dataset pequeño.
    """
    def __init__(self, input_dim = 10, output_dim = 2):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.output = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.dropout(x)
        x = self.activation(self.hidden2(x))
        return self.output(x)