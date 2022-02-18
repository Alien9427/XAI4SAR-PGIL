import math

def net_param_setting(model, init_lr):
    cls_list = list(map(id, model.cls_layer.parameters()))
    # comp_list = list(map(id, model.comp_layer.parameters())) + list(map(id, model.comp_fc.parameters()))
    # lat_list = list(map(id, model.latlayer1.parameters())) + list(map(id, model.latlayer2.parameters()))

    # base_params = filter(lambda p: id(p) not in cls_list + comp_list, model.parameters())
    base_params = filter(lambda p: id(p) not in cls_list, model.parameters())
    cls_params = filter(lambda p: id(p) in cls_list, model.parameters())
    # comp_params = filter(lambda p: id(p) in comp_list, model.parameters())
    # lat_params = filter(lambda p: id(p) in lat_list, model.parameters())

    param_list = [{'params': base_params, 'lr': init_lr * 0.0001},
                  {'params': cls_params, 'lr': init_lr * 0.5},
                  # {'params': lat_params, 'lr': init_lr * 0.5},
                  # {'params': comp_params, 'lr': init_lr * 0}
                  ]

    return param_list

def cosine_lr_schedule(t, n_t, T):
    """
    t: warming up
    T: total epochs
    n_t:
    """
    return lambda epoch: (0.9*epoch / t+0.1) if epoch < t else  0.1  \
        if n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))<0.1 else n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))