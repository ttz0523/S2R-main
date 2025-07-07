import torch

from network.blocks.dwt2d import DWTForward, DWTInverse

xfm = DWTForward(J=2, wave='db1', mode='zero')  # 做两次dwt变换
xfm2 = DWTForward(J=1, wave='db1', mode='zero')  # 做1次dwt变换
X = torch.randn(10,4,128,128)  # batch_size=10 通道为4，大小为128的二维图像
Yl, Yh = xfm(X)
Yl2, Yh2 = xfm2(X)
print(Yl.shape)  # 第二次dwt的低频1个子带
print(Yh[0].shape)  # 第一次dwt的高频三个子带LH, HL and HH
print(Yh[1].shape)  # 第二次dwt的高频三个子带LH, HL and HH
print(Yh[1][:,:,0].shape)    # 第二次dwt的高频子带LH，[:,:,0]表示取Yh[1]的第3个维度为0的其他所有值，因为0的位置在第三个，后面省略了两位[:,:,0,:,:]
# print(Yh[2].shape)
print(Yh2[0].shape)
print(Yh2[0][:,:,0].shape)
x = Yh2[0][:,:,0]
Yh2[0][:,:,0] = x
ifm = DWTInverse(wave='db1', mode='zero')  # 逆DWT
Y = ifm((Yl, Yh))