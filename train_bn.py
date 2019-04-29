import numpy as np

def get_ys(xs,w,b):
    # todo forward bn
    output0 = np.array([w[0]*x+b[0] for x in xs])
    bn0_mean = np.mean(output0)
    bn0_std = np.std(output0)
    output1 = np.array([w[1]*(out-bn0_mean)/float(bn0_std)+b[1] for out in output0])
    bn1_mean = np.mean(output1)
    bn1_std = np.std(output1)
    output2 = np.array([w[2]*(out-bn1_mean)/float(bn1_std)+b[2] for out in output1])

    bn_result = [
        [bn0_mean, bn0_std],
        [bn1_mean, bn1_std]
    ]
    # print(output2)
    # print(bn_result)
    return output2, bn_result

def get_ys_gt(xs,w,b):
    output0 = np.array([w[0]*x+b[0] for x in xs])
    output1 = np.array([w[1]*out+b[1] for out in output0])
    output2 = np.array([w[2]*(out)+b[2] for out in output1])
    return output2, None

def get_gradient_wo_bn(x,w,b,y,y_pred,bn_result):
    miu1, theta1 = bn_result[0]
    miu1 = float(miu1)
    theta1 = float(theta1)
    miu2, theta2 = bn_result[1]
    miu2 = float(miu2)
    theta2 = float(theta2)
    dy_dw = [
        w[2]*w[1]*x/theta1/theta2,
        w[0]*w[2]*x/theta1/theta2+w[2]*(b[0]-miu1)/theta1/theta2,
        w[1]*w[0]*x/theta1/theta2+w[2]*(b[0]-miu1)/theta1/theta2+(b[1]-miu2)/theta2
    ]
    grad_w = []
    for item in dy_dw:
        grad_w.append(float(y_pred-y)*item)
    dy_db = [
        w[2]*w[1]/theta1/theta2,
        w[2]/theta2,
        1.0
    ]
    grad_b = []
    for item in dy_db:
        grad_b.append(float(y_pred-y)*item)
    return np.array(grad_w), np.array(grad_b)

gt_w = np.array([2.0,2.0,2.0])
gt_b = np.array([1.0,1.0,1.0])

lr = 0.1
batch_x = [1.0,3.0,5.0,7.0]
test_x = [2.0,4.0,6.,8.0]

w = np.array([0.3246,0.982375,0.12531])
b = np.array([0.64325,0.75637,0.32545])

bn = []

for train_iter in  range(400):
    batch_y, bn_result_gt = get_ys_gt(batch_x,gt_w,gt_b)
    print(batch_y, bn_result_gt)
    batch_yhat, bn_result = get_ys(batch_x,w,b)
    print(batch_yhat, bn_result)
    
    batch_grad = []
    
    for x,y,yhat in zip(batch_x,batch_y,batch_yhat):
        print(x,y,yhat)
        batch_grad.append(get_gradient_wo_bn(x,w,b,y,yhat,bn_result))
    
    for grad in batch_grad:
        print("w:",w)
        print(grad[0])
        w = w - grad[0]*lr
        print("w':",w)
        b = b - grad[1]*lr
    print("epc:%d"%train_iter)
    print(batch_y)
    print(batch_yhat)
    print("***grad***")
    print(batch_grad[0])
    print(batch_grad[1])
    print("***weight***")
    print(w)
    print(b)
    print("***loss***")
    print(np.sum(np.abs(batch_y-batch_yhat)))

abs_error = 0
ys,_ = get_ys_gt(test_x,gt_w,gt_b)
ys_hat,_ = get_ys(test_x,w,b)
print(ys)
print(ys_hat)
abs_error = np.sum(np.abs(ys-ys_hat))
# for x in test_x:
#     y = get_ys(x,gt_w,gt_b)
#     y_hat = get_y(x,w,b)
#     abs_error+=np.abs(y-y_hat)
print("test_error:",abs_error)