from src.conv_onet import models


def get_model(cfg):
    """
    Return the network model.

    Args:
        cfg (dict): imported yaml config.
        nice (bool, optional): whether or not use Neural Implicit Scalable Encoding. Defaults to False.

    Returns:
        decoder (nn.module): the network model.
    """
    use_viewdirs = cfg['use_viewdirs']
    dim = cfg['data']['dim']
    coarse_grid_len = cfg['grid_len']['coarse']
    middle_grid_len = cfg['grid_len']['middle']
    fine_grid_len = cfg['grid_len']['fine']
    color_grid_len = cfg['grid_len']['color'] 
    c_dim = cfg['model']['c_dim']  # feature dimensions
    pos_embedding_method = cfg['model']['pos_embedding_method']

    decoder = models.decoder_dict['enerf'](
        dim=dim, c_dim=0, color=True,use_viewdirs=use_viewdirs,input_ch_views=93, 
        hidden_size=256, skips=[1], n_blocks=3, pos_embedding_method=pos_embedding_method
        )
    return decoder 
