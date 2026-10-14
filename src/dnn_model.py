#!/usr/bin/env python3
"""
Feedforward Binary DNN Model for DT049 Experiment V2
====================================================
Reference Architecture:
  Kuchar & Fujdiak (2025), "Analyzing anomalies in industrial networks:
  A data-driven approach to enhance security in manufacturing processes",
  Computers & Security, Vol. 153, 104395 (DOI: 10.1016/j.cose.2025.104395),
  Section 5.5 / Listing 1.

Verified Layer & Activation Structure (from Listing 1):
  Input(79)  [Adapted from paper input_dim=10; scaled via frozen StandardScaler]
    -> Linear(79, 100)  + ReLU
    -> Linear(100, 70)  + Tanh
    -> Linear(70, 100)  + ReLU
    -> Linear(100, 80)  + Tanh
    -> Linear(80, 60)   + Tanh
    -> Dropout(0.3)
    -> Linear(60, 10)   + Tanh
    -> Dropout(0.3)
    -> Linear(10, 1)    [1 output logit for BCEWithLogitsLoss; Sigmoid at inference]

Note on Paper Adaptations:
  - Paper Listing 1 starts with `model.add(norm_layer)` on `input_dim=10` and ends
    with `Dense(4, activation="sigmoid")`. Because `norm_layer` definition is not
    specified in Listing 1 and HAI 21.03 has 79 features with binary `attack` labels,
    DNN V2 uses the frozen offline StandardScaler (`models/preprocessing/scaler.joblib`)
    on 79 HAI features and outputs 1 binary logit trained with BCEWithLogitsLoss.
  - Expected trainable parameters: 35,731.
  - Strictly no BatchNorm or LayerNorm layers are added.
"""

from typing import Dict, List, Sequence, Tuple
import torch
import torch.nn as nn


REFERENCE_HIDDEN_DIMS: Tuple[int, ...] = (100, 70, 100, 80, 60, 10)
VERIFIED_HIDDEN_ACTIVATIONS: Tuple[str, ...] = ("relu", "tanh", "relu", "tanh", "tanh", "tanh")
VERIFIED_ACTIVATION_STATUS: str = "VERIFIED_FROM_PAPER_LISTING_1"
REFERENCE_DROPOUT_RATE: float = 0.3
EXPECTED_TRAINABLE_PARAMETERS: int = 35731
SUPPORTED_ACTIVATIONS = frozenset({"relu", "tanh"})


def _make_activation(name: str) -> nn.Module:
    norm_name = str(name).strip().lower()
    if norm_name == "relu":
        return nn.ReLU()
    if norm_name == "tanh":
        return nn.Tanh()
    raise ValueError(
        f"Unsupported activation '{name}'. Allowed values from Listing 1 are {sorted(SUPPORTED_ACTIVATIONS)}."
    )


class FeedforwardDNNV2(nn.Module):
    """
    Feedforward Binary DNN adapted from Kuchar & Fujdiak (2025) Section 5.5 / Listing 1.
    """

    def __init__(
        self,
        hidden_activations: Sequence[str] = VERIFIED_HIDDEN_ACTIVATIONS,
        input_size: int = 79,
        hidden_dims: Sequence[int] = REFERENCE_HIDDEN_DIMS,
        dropout_rate: float = REFERENCE_DROPOUT_RATE,
        output_size: int = 1,
        activation_status: str = VERIFIED_ACTIVATION_STATUS,
    ):
        super().__init__()
        if int(input_size) != 79:
            raise ValueError(f"Expected input_size=79 for HAI 21.03, got {input_size}")
        if tuple(hidden_dims) != REFERENCE_HIDDEN_DIMS:
            raise ValueError(
                f"Hidden layer dimensions must match paper reference {REFERENCE_HIDDEN_DIMS}, "
                f"got {tuple(hidden_dims)}"
            )
        if len(hidden_activations) != len(REFERENCE_HIDDEN_DIMS):
            raise ValueError(
                f"Expected exactly {len(REFERENCE_HIDDEN_DIMS)} hidden activations for layers "
                f"{REFERENCE_HIDDEN_DIMS}, got {len(hidden_activations)}"
            )
        if abs(float(dropout_rate) - REFERENCE_DROPOUT_RATE) > 1e-9:
            raise ValueError(
                f"Dropout rate must match paper reference {REFERENCE_DROPOUT_RATE}, got {dropout_rate}"
            )
        if int(output_size) != 1:
            raise ValueError(f"Binary V2 DNN requires output_size=1 logit, got {output_size}")

        self.input_size = int(input_size)
        self.hidden_dims: Tuple[int, ...] = tuple(int(d) for d in hidden_dims)
        self.hidden_activations: Tuple[str, ...] = tuple(str(a).strip().lower() for a in hidden_activations)
        self.dropout_rate = float(dropout_rate)
        self.output_size = int(output_size)
        self.activation_status = str(activation_status)

        # Explicit layer construction matching Section 5.5 / Listing 1:
        # 1. Dense(100) -> relu
        # 2. Dense(70)  -> tanh
        # 3. Dense(100) -> relu
        # 4. Dense(80)  -> tanh
        # 5. Dense(60)  -> tanh -> Dropout(0.3)
        # 6. Dense(10)  -> tanh -> Dropout(0.3)
        # 7. Output(1 logit)
        self.fc1 = nn.Linear(79, 100)
        self.act1 = _make_activation(self.hidden_activations[0])

        self.fc2 = nn.Linear(100, 70)
        self.act2 = _make_activation(self.hidden_activations[1])

        self.fc3 = nn.Linear(70, 100)
        self.act3 = _make_activation(self.hidden_activations[2])

        self.fc4 = nn.Linear(100, 80)
        self.act4 = _make_activation(self.hidden_activations[3])

        self.fc5 = nn.Linear(80, 60)
        self.act5 = _make_activation(self.hidden_activations[4])
        self.drop1 = nn.Dropout(p=self.dropout_rate)

        self.fc6 = nn.Linear(60, 10)
        self.act6 = _make_activation(self.hidden_activations[5])
        self.drop2 = nn.Dropout(p=self.dropout_rate)

        self.out = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw logits of shape (batch_size, 1).
        """
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        h = self.act3(self.fc3(h))
        h = self.act4(self.fc4(h))
        h = self.drop1(self.act5(self.fc5(h)))
        h = self.drop2(self.act6(self.fc6(h)))
        logits = self.out(h)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Sigmoid at inference time to return P(attack=1 | x) in (0, 1)
        of shape (batch_size, 1).
        """
        return torch.sigmoid(self.forward(x))


def count_trainable_parameters(model: nn.Module) -> int:
    """Returns exact count of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_layer_parameter_breakdown(model: nn.Module) -> List[Dict]:
    """Returns parameter shapes and counts for each named parameter tensor."""
    breakdown = []
    for name, param in model.named_parameters():
        breakdown.append({
            "name": name,
            "shape": list(param.shape),
            "numel": int(param.numel()),
            "requires_grad": bool(param.requires_grad),
        })
    return breakdown


def verify_dnn_v2_architecture(
    model: FeedforwardDNNV2,
    expected_params: int = EXPECTED_TRAINABLE_PARAMETERS,
) -> Dict:
    """
    Audits a FeedforwardDNNV2 instance to confirm:
    1. Total trainable parameters == 35,731.
    2. Zero BatchNorm or LayerNorm modules.
    3. Exact two Dropout(0.3) layers placed after fc5+act5 and fc6+act6.
    4. Exact 7 Linear layers (6 hidden + 1 output).
    """
    total_params = count_trainable_parameters(model)
    if total_params != expected_params:
        raise ValueError(
            f"Parameter count mismatch: expected {expected_params}, got {total_params}"
        )

    norm_modules = [
        m for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm))
    ]
    if norm_modules:
        raise ValueError(f"Forbidden normalization layers found in model: {norm_modules}")

    dropout_modules = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    if len(dropout_modules) != 2 or any(abs(m.p - REFERENCE_DROPOUT_RATE) > 1e-9 for m in dropout_modules):
        raise ValueError(f"Expected exactly 2 Dropout(0.3) layers, got {dropout_modules}")

    linear_modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
    expected_pairs = [(79, 100), (100, 70), (70, 100), (100, 80), (80, 60), (60, 10), (10, 1)]
    actual_pairs = [(m.in_features, m.out_features) for m in linear_modules]
    if actual_pairs != expected_pairs:
        raise ValueError(f"Linear layer dimensions mismatch: expected {expected_pairs}, got {actual_pairs}")

    return {
        "trainable_parameters": total_params,
        "linear_layer_dimensions": actual_pairs,
        "hidden_activations": list(model.hidden_activations),
        "activation_status": model.activation_status,
        "dropout_count": len(dropout_modules),
        "dropout_rate": REFERENCE_DROPOUT_RATE,
        "normalization_layers_count": len(norm_modules),
        "parameter_breakdown": get_layer_parameter_breakdown(model),
    }
