from transcription.device_configuration import DeviceConfiguration


# TODO: implement saving & removing configuration
def save_configuration(cfg: DeviceConfiguration):
    config = {
        "Model": cfg.model_name,
        "Batch Size": cfg.batch_size,
        "Data Type": cfg.data_type,
    }


def load_config():
    pass


def delete_config():
    pass
