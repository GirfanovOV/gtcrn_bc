import torch


def component_grad_norms(model, grad_components):
    params = [p for p in model.parameters() if p.requires_grad]
    grads_by_name = {}
    norms = {}
    for name, component in grad_components.items():
        grads = torch.autograd.grad(
            component,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        grads_by_name[name] = grads
        sq_norm = torch.zeros((), device=component.device)
        for grad in grads:
            if grad is not None:
                sq_norm = sq_norm + grad.detach().pow(2).sum()
        norms[name] = sq_norm.sqrt()

    metrics = {
        f"grad_norm_{name}": norm.item()
        for name, norm in norms.items()
    }
    sisnr_grads = grads_by_name["weighted_sisnr"]
    sisnr_norm = norms["weighted_sisnr"]
    for target, short_name in (
        ("weighted_real", "real"),
        ("weighted_imag", "imag"),
        ("weighted_mag", "mag"),
        ("weighted_spectral", "spectral"),
    ):
        dot = torch.zeros((), device=sisnr_norm.device)
        for sisnr_grad, target_grad in zip(sisnr_grads, grads_by_name[target]):
            if sisnr_grad is not None and target_grad is not None:
                dot = dot + (sisnr_grad.detach() * target_grad.detach()).sum()
        denom = sisnr_norm * norms[target] + 1e-12
        metrics[f"cos_sisnr_{short_name}"] = (dot / denom).item()
    return metrics
