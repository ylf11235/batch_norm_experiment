import numpy as np

def get_y(x,w,b):
    return w[2]*(w[1]*(w[0]*x+b[0])+b[1])+b[2]

def get_gradient_wo_bn(x,w,b,y,y_pred,if_loss_norm=False):
    dy_dw = [
        w[2]*w[1]*x,
        w[0]*w[2]*x+w[2]*b[0],
        w[1]*w[0]*x+w[2]*b[0]+b[1]
    ]
    grad_w = []
    for item in dy_dw:
        grad_w.append(float(y_pred-y)*item)
    grad_w = np.array(grad_w)
    dy_db = [
        w[2]*w[1],
        w[2],
        1.0
    ]
    grad_b = []
    for item in dy_db:
        grad_b.append(float(y_pred-y)*item)
    grad_b = np.array(grad_b)
    if if_loss_norm:
        v_grad_len = (np.sum(grad_w*grad_w)+np.sum(grad_b*grad_b))**0.5
        # print(v_grad_len)
        # raise
        grad_w = grad_w / v_grad_len
        grad_b = grad_b / v_grad_len
    else:
        v_grad_len = 1
    return np.array(grad_w), np.array(grad_b), v_grad_len

gt_w = np.array([2.0,2.0,2.0])
gt_b = np.array([1.0,1.0,1.0])

lr = 0.0001
batch_x = [1.0,3.0,5.0,7.0]
test_x = [2.0,4.0,6.0]

w = np.array([0.3246,0.982375,0.12531])
b = np.array([0.64325,0.75637,0.32545])

loss_norm = True

lrs = np.arange(0.005,0.025,0.001)
fake_lrs = []
test_errors = []
lr = 0.007
# for lr in lrs:
for train_iter in  range(400):
    batch_y = []
    for x in batch_x:
        batch_y.append(get_y(x,gt_w,gt_b))
    batch_yhat = []
    for x in batch_x:
        batch_yhat.append(get_y(x,w,b))
    batch_grad = []
    for x,y,yhat in zip(batch_x,batch_y,batch_yhat):
        batch_grad.append(get_gradient_wo_bn(x,w,b,y,yhat,if_loss_norm=loss_norm))
    
    avg_grad_len = 0
    for grad in batch_grad:
        w = w - grad[0]*lr
        # b = b - grad[1]*lr
        grad_len = grad[2]
        avg_grad_len+=grad_len
    avg_grad_len/=4.0
    fake_lr = lr / float(avg_grad_len)
    fake_lrs.append(fake_lr)
    # print("epc:%d"%train_iter)
    # print(batch_y)
    # print(batch_yhat)
    # print("***grad***")
    # print(batch_grad[0])
    # print(batch_grad[1])
    # print("***weight***")
    # print(w)
    # print(b)

abs_error = 0
for x in test_x:
    y = get_y(x,gt_w,gt_b)
    y_hat = get_y(x,w,b)
    abs_error+=np.abs(y-y_hat)
print("test_error:",abs_error)
test_errors.append(abs_error)

print(lrs)
print(fake_lrs)
# print(test_errors)