import numpy as np


#hàm nhận giá trị đầu vào trả về đầu ra trong khoảng (0,,1)
def sigmoid(z) :
    return 1./(1. + np.exp(-z))
#chuyển đổi label thành mã one hot y = nhãn , num_label = số lượng lớp
def int_to_onehot(y,num_labels) :
    ary  = np.zeros((y.shape[0],num_labels))
    for i,val in enumerate(y) :
        ary[i,val] = 1
    return ary

class NeuralNetMLP :
    #khởi tạo trọng số, bias, số lượng đầu vào,lớp ẩn,lớp ra 
    def __init__(self,num_features,num_hidden, num_classes, random_seed = 123) :
        super().__init__()
        
        self.num_classes = num_classes
        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(loc = 0.0, scale=0.1,size = (num_hidden,num_features))
        self.bias_h = np.zeros(num_hidden)
        
        self.weight_out = rng.normal(loc = 0.0, scale=0.1,size = (num_classes,num_hidden))
        self.bias_out = np.zeros(num_classes)
    #hàm lan truyền tiến
    def forward(self,x) :
        #tính toán giá trị lớp ẩn bằng dữ liệu đầu vào
        #a_h đầu ra của lớp ẩn
        z_h = np.dot(x,self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)
        
        #tính toán giá trị đàu ra bằng gữ liệu lớp ẩn
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out
    
    #hàm truyền ngược(tính gradient và cập nhập trọng số)
    def backward(self,x,a_h,a_out,y) :
        
        y_onehot = int_to_onehot(y,self.num_classes)
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1. - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        d_a_h__d_z_h = a_h * (1. - a_h)
        d_z_h__d_w_h = x
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T,d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)
        return (d_loss__dw_out, d_loss__db_out,d_loss__d_w_h, d_loss__d_b_h)


