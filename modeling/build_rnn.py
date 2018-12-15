import torch
import torch.nn as nn
import numpy as np

class build_rnn(nn.Module):
    def __init__(self, in_dim,device):
        super(SCNN, self).__init__()
        self.device = device
        #nn.Conv2d = [in_channels, out_channels, kernel_size, stride, padding]
        self.conv1 = nn.Conv2d(1,1,kernel_size=(1,9),padding=(0,4))
        # torch.nn.functional.pad()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=(1, 9), padding=(0, 4))
        # self.conv = nn.Conv2d(256,256)
    def forward(self,x ):

        # top to down #
        feature_list_old = []
        feature_list_new = []
        for cnt in range(x.size()[1]):
            feature_list_old.append(np.expand_dims(x[:, cnt, :, :].cpu().detach().numpy(), axis=1))

        feature_list_new.append(np.expand_dims(x[:, 0, :, :].cpu().detach().numpy(), axis=1))
        top_2_down = self.conv1(torch.from_numpy(feature_list_old[0]).float().to(self.device))
        top_2_down = self.relu(top_2_down)
        top_2_down = torch.add(top_2_down,torch.from_numpy(feature_list_old[0]).float().to(self.device))
        feature_list_new.append(top_2_down)
        for cnt in range(2, x.size()[1]):
            top_2_down = self.relu(self.conv1(feature_list_new[cnt-1]))
            top_2_down = torch.add(torch.from_numpy(feature_list_old[cnt]).float().to(self.device), top_2_down)
            feature_list_new.append(top_2_down)

        # down to top #
        feature_list_old = feature_list_new
        feature_list_new = []
        feature_len = len(feature_list_old)-1
        # print("feature_len:",feature_len)
        feature_list_new.append(feature_list_old[feature_len])

        down_2_top = torch.add(self.relu(self.conv1(feature_list_old[feature_len])),
                               feature_list_old[feature_len-1])
        feature_list_new.append(down_2_top)

        for cnt in range(2, x.size()[1]):
            down_2_top = self.relu(self.conv1(feature_list_new[cnt - 1]))
            # print(cnt)
            # print("down_2_top:",type(down_2_top))
            # print("feature_list_old[feature_len - cnt]:",type(feature_list_old[feature_len - cnt]))
            # down_2_top = torch.add(down_2_top,feature_list_old[feature_len-cnt])
            try:
                down_2_top = torch.add(feature_list_old[feature_len - cnt],down_2_top)
            except:
                down_2_top = torch.add(torch.from_numpy(feature_list_old[feature_len - cnt]).float().to(self.device), down_2_top)
            feature_list_new.append(down_2_top)

        feature_list_new.reverse()
        processed_feature = torch.cat(feature_list_new, dim=1)
        processed_feature = processed_feature.squeeze()
        # left to right #

        feature_list_old = []
        feature_list_new = []
        for cnt in range(processed_feature.size()[2]):
            feature_list_old.append(np.expand_dims(processed_feature[:, :, cnt, :].cpu().detach().numpy(), axis=2))
        feature_list_new.append(np.expand_dims(processed_feature[:, :, 0, :].cpu().detach().numpy(), axis=2))

        left_2_right = torch.add(self.relu(self.conv2(torch.from_numpy(feature_list_old[0]).float().to(self.device))),
                                 torch.from_numpy(feature_list_old[1]).float().to(self.device))
        feature_list_new.append(left_2_right)
        # print()processed_feature.size()[2]
        for cnt in range(2, processed_feature.size()[2]):
            # with tf.variable_scope("convs_6_3", reuse=True):
            left_2_right = torch.add(self.relu(self.conv2(feature_list_new[cnt - 1])),
                                     torch.from_numpy(feature_list_old[cnt]).float().to(self.device))
            feature_list_new.append(left_2_right)
        # right to left #

        feature_list_old = feature_list_new
        feature_list_new = []
        feature_list_len = len(feature_list_old)-1
        feature_list_new.append(feature_list_old[feature_list_len])

        # w4 = tf.get_variable('W4', [9, 1, 128, 128],
        #                      initializer=tf.random_normal_initializer(0, math.sqrt(2.0 / (9 * 128 * 128 * 5))))
        # with tf.variable_scope("convs_6_4"):
        feature_list_old_len = len(feature_list_old)-1
        # print(feature_list_old_len)
        # print(feature_list_old_len)
        conv_6_4 = torch.add(self.relu(self.conv2(feature_list_old[feature_list_old_len])),
                          feature_list_old[feature_list_old_len-1])
        feature_list_new.append(conv_6_4)

        for cnt in range(2, processed_feature.size()[2]):
            # with tf.variable_scope("convs_6_4", reuse=True):
            try:
                conv_6_4 = torch.add(self.relu(self.conv2(feature_list_new[cnt - 1])),
                                 feature_list_old[feature_list_old_len - cnt])
            except:
                conv_6_4 = torch.add(self.relu(self.conv2(feature_list_new[cnt - 1])),
                                     torch.from_numpy(feature_list_old[feature_list_old_len - cnt]).float().to(self.device))
            feature_list_new.append(conv_6_4)

        feature_list_new.reverse()
        processed_feature = torch.cat(feature_list_new, dim=2)
        processed_feature = processed_feature.squeeze()

        output = processed_feature.view(processed_feature.size()).permute(0,3, 1 ,2)
        # print(output.size())

        return output
        #######################
if __name__ == "__main__":

    inputs = torch.rand(8, 128, 36, 100)# N x C x H x W
    # N, C, H, W = x.size()
    inputs = inputs.view(inputs.size()).permute(0, 2, 3, 1)
    print("x.size():", inputs.size())


    # inputs = torch.rand(8, 36, 100, 128)# N x H x W x C
    channel = inputs.size()[1]
    model = build_rnn(channel)
    output = model(inputs)
    print(output.size())
