"""Explicit fixed ridge probe; no hidden-weight learning or backward pass."""


def conjugate_gradient(xp, matrix, right, max_iterations=512, tolerance=1e-9):
    value = xp.zeros_like(right)
    residual = right.copy()
    direction = residual.copy()
    rr = xp.sum(residual * residual)
    norm = float(xp.sqrt(rr))
    if norm == 0:
        return value, {"iterations": 0, "relative_residual": 0.0}
    for iteration in range(1, max_iterations + 1):
        product = matrix @ direction
        denominator = xp.sum(direction * product)
        if float(denominator) <= 0:
            raise ArithmeticError("Ridge system is not positive definite")
        alpha = rr / denominator
        value += alpha * direction
        residual -= alpha * product
        next_rr = xp.sum(residual * residual)
        if float(xp.sqrt(next_rr)) <= tolerance * norm:
            measured = float(xp.sqrt(xp.sum((matrix @ value - right) ** 2))) / norm
            if measured > tolerance * 1.01:
                raise ArithmeticError("CG recursive/actual residual disagreement")
            return value, {"iterations": iteration, "relative_residual": measured}
        direction = residual + (next_rr / rr) * direction
        rr = next_rr
    raise ArithmeticError("Ridge probe solver did not converge within fixed budget")


def fit_probe(xp, training_features, targets):
    x = xp.asarray(training_features, dtype=xp.float64)
    y = xp.asarray(targets, dtype=xp.float64) * 2 - 1
    mean = x.mean(axis=0)
    scale = xp.maximum(x.std(axis=0), 1e-3)
    design = xp.concatenate(((x - mean) / scale, xp.ones((len(x), 1))), axis=1)
    gram = design.T @ design + xp.eye(design.shape[1], dtype=xp.float64)
    right = design.T @ y
    weights, solver = conjugate_gradient(xp, gram, right)
    return {"weights": weights, "mean": mean, "scale": scale, "solver": solver,
            "gram": gram, "right": right}


def predict_probe(xp, features, probe):
    x = xp.asarray(features, dtype=xp.float64)
    scores = ((x - probe["mean"]) / probe["scale"]) @ probe["weights"][:-1] + probe["weights"][-1]
    return scores, (scores > 0).astype(xp.int64)
