import torch


def active_boundary_loss(y_true, y_pred, weight=10):
    '''
    y_true, y_pred: tensor of shape (B, C, H, W, D), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    '''
    # length term
    delta_x = y_pred[:, :, 1:, :, :] - y_pred[:, :,
                                              :-1, :, :]  # x gradient (B, C, H-1, W, D)
    delta_y = y_pred[:, :, :, 1:, :] - y_pred[:, :,
                                              :, :-1, :]  # y gradient   (B, C, H,   W-1, D)
    delta_z = y_pred[:, :, :, :, 1:] - y_pred[:, :,
                                              :, :, :-1]  # z gradient   (B, C, H,   W, D-1)

    delta_x = delta_x[:, :, 1:, :-2, :-2]**2  # (B, C, H-2, W-2, D-2)
    delta_y = delta_y[:, :, :-2, 1:, :-2]**2  # (B, C, H-2, W-2, D-2)
    delta_z = delta_z[:, :, :-2, :-2, 1:]**2  # (B, C, H-2, W-2, D-2)

    delta_pred = torch.abs(delta_x + delta_y + delta_z)

    # where is a parameter to avoid square root is zero in practice.
    epsilon = 1e-8
    # eq.(11) in the paper, mean is used instead of sum.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))

    # region term
    # import pdb;pdb.set_trace()
    loss = 0
    print(y_pred.max(), y_pred.min())
    try:
        pred_classes = y_pred.unique()
    except:
        import pdb;pdb.set_trace()
    if len(pred_classes) == 1: #只有背景
        loss = weight*lenth
    else:

        for pred_class in pred_classes[1:]:
            y_cp = y_pred.clone()
            y_cp[y_pred==pred_class] = 1
            y_cp[y_pred!=pred_class] = 0
            c_in = torch.sum(y_cp * y_true) / torch.sum(y_cp) # 内部均值
            c_out = torch.sum((1-y_cp) * y_true) / torch.sum(1-y_cp) # 外部均值

            # equ.(12) in the paper, mean is used instead of sum.
            region_in = torch.mean(y_cp * (y_true - c_in)**2)
            # region_out = torch.mean((1-y_cp) * (y_true - c_out)**2)
            # region = region_in + region_out

            # loss =  weight*lenth + region
            loss += region_in
        loss += weight*lenth
    return loss
