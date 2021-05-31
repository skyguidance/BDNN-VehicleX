from collections import OrderedDict


def sync_weights(config, model_F, model_R, is_F_to_R):
    """
    BDNN model weights sync
    :param config: config
    :param model_F: Stage-F forward model
    :param model_R: Stage-R backward model
    :param is_F_to_R: True if sync model weights from F to R. False if R to F.
    """
    if is_F_to_R:
        config["logger"].info("Syncing Weights F -> R ...")
        state_dict = model_F.state_dict()
        transformed_state_dict = OrderedDict()
        max_mlp_layer_index = len(config["model"]["BDNN"]["mlp_layers"]) - 1
        for k, v in state_dict.items():
            # Copy BN
            if "bn" in k:
                transformed_state_dict[k] = v
            # Copy Weights
            elif ("weight" in k) and (len(v.shape) == 2):
                transformed_state_dict[k] = v.T
            # Copy Bias
            elif ("bias" in k) and (f"layer{max_mlp_layer_index}" in k):
                transformed_state_dict[f"net.final.bias"] = v
            elif ("bias" in k) and ("final" not in k):
                current_layer_index = int(k.split(".")[1].replace("layer", ""))
                transformed_state_dict[f"net.layer{current_layer_index + 1}.bias"] = v

        model_R.load_state_dict(transformed_state_dict, strict=False)
    else:
        config["logger"].info("Syncing Weights R -> F ...")
        state_dict = model_R.state_dict()
        transformed_state_dict = OrderedDict()
        max_mlp_layer_index = len(config["model"]["BDNN"]["mlp_layers"]) - 1
        for k, v in state_dict.items():
            # Copy BN
            if "bn" in k:
                transformed_state_dict[k] = v
            # Copy Weights
            elif ("weight" in k) and (len(v.shape) == 2):
                transformed_state_dict[k] = v.T
            # Copy Bias
            elif ("bias" in k) and ("final" in k):
                transformed_state_dict[f"net.layer{max_mlp_layer_index}.bias"] = v
            elif ("bias" in k) and ("layer0" not in k):
                current_layer_index = int(k.split(".")[1].replace("layer", ""))
                transformed_state_dict[f"net.layer{current_layer_index - 1}.bias"] = v
        model_F.load_state_dict(transformed_state_dict, strict=False)
