import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class gwnet(nn.Module):
    def __init__(self,no_pmt,prompt_dim,device, remove_list, add_list, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.no_pmt = no_pmt
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        #self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1)) #1,1->1,3
        self.supports = supports

        receptive_field = 1

        # 去掉节点
        # 声明模型时给出去掉的节点列表remove_list
        self.remove_list = remove_list
        self.mask = torch.ones(num_nodes,num_nodes).to(device)
        if self.remove_list is not None:
            for i in self.remove_list:
                self.mask[i]=0
                self.mask[:,i]=0

        self.add_list = add_list
        if self.add_list is not None:
            for i in self.add_list:
                self.mask[i]=0
                self.mask[:,i]=0
        self.supports_len = 0
        self.pemb_multi_t = MultiScaleTemporalAutoencoder()
        self.pemb_multi_s = HierarchicalSpatialAutoencoder(num_nodes)
        self.pemb_s = nn.Linear(128, 13)
        self.pemb_t = nn.Linear(384, 13)
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1




        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field


 
    def forward(self, original_input,prompt=0,add_test=0,remove_test=0):
        in_len = original_input.size(3)
        input = nn.functional.pad(original_input[:,:,:,:12], (1,0,0,0))

        # prompt -- use_prompt; no_pmt -- ablation_pmt
        if prompt != 0 and self.no_pmt != 1:
            # prompt = original_input[:,:,:,12:]
            # input of prompt_autoencoder for reconstruction
            prompt_input = original_input[:, :, :, :12].squeeze()
            # _, prompt_t, prompt_s = self.pemb(prompt_input)
            a, prompt_t = self.pemb_multi_t(prompt_input)
            b, prompt_s = self.pemb_multi_s(prompt_input)
            prompt_t = prompt_t.unsqueeze(1)
            prompt_s = prompt_s.unsqueeze(1)
            prompt_t = torch.tanh(self.pemb_t(prompt_t))
            prompt_s = torch.tanh(self.pemb_s(prompt_s))
            # prompt + input
            input = prompt_t + prompt_s + input
        if in_len<self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0
        
        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)*self.mask
            # 如果是去除节点任务
            if self.remove_list is not None:
                # 去除节点
                adp = self.mask * adp
                # 如果是测试阶段
                if remove_test == 1:
                    for i in self.remove_list:
                        max_indices = max_similarity(x) # b t n c
                        k = max_indices[i]
                        adp[i]=adp[k]
                        adp[:,i]=adp[:,k]
            #print(self.add_list,self.add_test)
            if self.add_list is not None and add_test>0:
                adp = self.mask * adp
            #    print(self.mask,adp)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip


            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x,self.supports)

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


def max_similarity(data_tensor):
    numerator = torch.einsum('btlc,btnc->btln', data_tensor, data_tensor)
    averaged_similarity = numerator.mean(dim=[0,1])
    max_indices = torch.argmax(averaged_similarity, dim=1)
    return max_indices

class MLP(nn.Module):
    def __init__(self, input_size, output_size,prompt_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, prompt_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(prompt_dim, output_size)
    def forward(self, x):
        x = self.fc(x)
        prompt = self.relu(x)
        x = self.fc2(prompt)
        return x,prompt


class PromptAutoencoder(nn.Module):
    def __init__(self, input_size=12, hidden_size=144, output_size=12):
        super(PromptAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size // 2),  # You can adjust the size of the encoded representation
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # x -> encoder -> embedding
        encoded = self.encoder(x)  # Get the encoded representation
        # embedding -> decoder -> x' -- reconstruction
        decoded = self.decoder(encoded)  # Decode the representation back to the original size
        encoded_t, encoded_s = encoded, encoded
        return decoded, encoded_t, encoded_s


class MultiScaleTemporalAutoencoder(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_scales=3):
        super(MultiScaleTemporalAutoencoder, self).__init__()

        # Encoding layers (Dilated Convolutions)
        self.encoders = nn.ModuleList([
            nn.Conv1d(input_size, hidden_size, kernel_size=3, dilation=2 ** i, padding=2 ** i)
            for i in range(num_scales)
        ])

        # Decoding layers (Reverse Dilated Convolutions)
        self.decoders = nn.ModuleList([
            nn.ConvTranspose1d(hidden_size, input_size, kernel_size=3, dilation=2 ** i, padding=2 ** i)
            for i in range(num_scales)
        ])

        # Fusion layer to combine multi-scale features
        # self.fusion = nn.Conv1d(num_scales * hidden_size, hidden_size, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, input_size, node)

        # Multi-scale encoding
        scale_features = []
        for encoder in self.encoders:
            scale_features.append(F.relu(encoder(x)))

        # Concatenate multi-scale features
        fused_features = torch.cat(scale_features, dim=1)
        # fused_features = F.relu(self.fusion(fused_features))

        # Multi-scale decoding
        reconstructed = 0
        for decoder, scale_feature in zip(self.decoders, scale_features):
            reconstructed += F.relu(decoder(scale_feature))

        # After decoding, we need to transpose back to (batch, node, input_size)
        reconstructed = reconstructed.transpose(1, 2)  # (batch, node, input_size)
        fused_features = fused_features.transpose(1, 2)

        return reconstructed, fused_features


class HierarchicalSpatialAutoencoder(nn.Module):
    def __init__(self, num_nodes, time_steps=12, latent_dim=128, num_classes=[64, 8]):
        super(HierarchicalSpatialAutoencoder, self).__init__()
        self.num_nodes = num_nodes
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.temperature = 0.5

        # 初始化Autoencoder
        self.fc = nn.Linear(time_steps, latent_dim)

    def forward(self, x):
        # 第一层 autoencoder
        emb = self.fc(x)
        aggregated_1 = self.aggregate_nodes(emb, 0, self.num_nodes)

        # 第二层 autoencoder
        aggregated_2 = self.aggregate_nodes(aggregated_1, 1, 64)

        # 第三层 autoencoder
        batch_size, num_clusters, embedding_dim = aggregated_2.shape
        # 展平为 (batch_size * num_clusters, embedding_dim)
        embeddings = aggregated_2.view(batch_size * num_clusters, embedding_dim)
        # 计算相似度矩阵
        similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        # 创建标签矩阵：相同类别的节点相似度为正样本，其他为负样本
        labels = torch.eye(batch_size * num_clusters).to(embeddings.device)
        # 计算对比损失
        logits = similarity_matrix / self.temperature
        labels = labels.float()
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        return torch.mean(loss), emb

    def aggregate_nodes(self, x, level, node_bef):
        """
        聚合节点基于时间变化的方差。
        :param x: 原始数据 (batch_size, num_nodes, time_steps)
        :return: 聚合后的节点 (batch_size, num_aggregated_nodes, time_steps)
        """
        # 计算每个节点在时间维度上的方差
        variances = torch.var(x, dim=2)  # (batch_size, num_nodes)

        # 对节点按照方差排序
        sorted_indices = torch.argsort(variances, dim=-1, descending=True)  # (batch_size, num_nodes)

        # 按方差大小均匀划分n类
        num_per_class = node_bef // self.num_classes[level]
        aggregated_nodes = []

        for i in range(self.num_classes[level]):
            start_idx = i * num_per_class
            end_idx = (i + 1) * num_per_class if i != self.num_classes[level] - 1 else node_bef
            class_indices = sorted_indices[:, start_idx:end_idx]

            # 聚合这些节点
            aggregated_class_nodes = torch.mean(
                x.gather(1, class_indices.unsqueeze(-1).expand(-1, -1, self.time_steps)), dim=1)
            aggregated_nodes.append(aggregated_class_nodes)

        return torch.stack(aggregated_nodes, dim=1)  # (batch_size, num_classes, time_steps)

