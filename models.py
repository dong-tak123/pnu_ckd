import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(num_features=hidden_size1)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_size2)

    def forward(self, x):
        # breakpoint()
        x = self.fc1(x)
        x = self.bn1(self.relu1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(self.relu2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# Simple MLP
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # breakpoint()
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.fc3(x)
        x = self.sigmoid(x)
        return x
    

# Grouping All
class GroupingMLP(nn.Module):
    def __init__(self, basic_dim, weight_dim, group_dim, hidden_dim, output_size):
        super(GroupingMLP, self).__init__()
        self.weight_fc = nn.Linear(weight_dim, group_dim)
        self.relu1 = nn.ReLU()
        self.total_fc1 = nn.Linear(group_dim + basic_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.total_fc2 = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, basic_x, weight_x):
        # breakpoint()
        grouped_weight = self.relu1(self.weight_fc(weight_x))
        mixed_ftr = torch.cat((basic_x, grouped_weight), dim=1)
        x = self.relu2(self.total_fc1(mixed_ftr))
        x = self.sigmoid(self.total_fc2(x))
        return x

group_dict = {
    "Rice" : ["CT1_W001", "CT1_W011", "CT1_W012"],
    "Other grains" : ["CT1_W002", "CT1_W003", "CT1_W004", "CT1_W005", "CT1_W020"],
    "Noodles and dumplings" : ["CT1_W006", "CT1_W007", "CT1_W008", "CT1_W009", "CT1_W010"],
    "Wheat flour and bread" : ["CT1_W013", "CT1_W014", "CT1_W016", "CT1_W017", "CT1_W018", "CT1_W019", "CT1_W021"],
    "Potatoes" : ["CT1_W028", "CT1_W029", "CT1_W030", "CT1_W031"], 
    "Sweets" : ["CT1_W015", "CT1_W022", "CT1_W090"], 
    "Soybean pastes" : ["CT1_W025"], 
    "Bean, tofu, and soymilk" : ["CT1_W024", "CT1_W027", "CT1_W088"], 
    "Nuts" : ["CT1_W023"], 
    "Vegetables" : ["CT1_W037", "CT1_W038", "CT1_W039", "CT1_W040", "CT1_W041", "CT1_W042", "CT1_W043", "CT1_W044", "CT1_W045", "CT1_W046", "CT1_W049", "CT1_W050",
                  "CT1_W051", "CT1_W052", "CT1_W053", "CT1_W054", "CT1_W055", "CT1_W056"], 
    "Kimchi" : ["CT1_W032", "CT1_W033", "CT1_W034", "CT1_W035", "CT1_W036"], 
    "Mushroom" : ["CT1_W047", "CT1_W048"], 
    "Fruits" : ["CT1_W095", "CT1_W096", "CT1_W097", "CT1_W098", "CT1_W099", "CT1_W100", "CT1_W101", "CT1_W102", "CT1_W103", "CT1_W104", "CT1_W105", "CT1_W106"], 
    "Red meat and its products" : ["CT1_W057", "CT1_W058", "CT1_W059", "CT1_W060", "CT1_W061", "CT1_W062", "CT1_W063", "CT1_W065", "CT1_W066"], 
    "White meat and its products" : ["CT1_W064"], 
    "Eggs" : ["CT1_W026"], 
    "Fish and shellfish" : ["CT1_W067", "CT1_W068", "CT1_W069", "CT1_W070", "CT1_W071", "CT1_W072", "CT1_W073", "CT1_W074", "CT1_W075", 
                          "CT1_W076", "CT1_W077", "CT1_W078", "CT1_W079", "CT1_W080", "CT1_W081"], 
    "Seaweeds" : ["CT1_W082", "CT1_W083"], 
    "Milk and dairy products" : ["CT1_W084", "CT1_W085", "CT1_W086", "CT1_W087", "CT1_W091"], 
    "Beverage" : ["CT1_W093", "CT1_W094"], 
    "Coffee and tea" : ["CT1_W089", "CT1_W092"]
}

class Linear_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Linear_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, self.out_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, data):
        return self.sigmoid(self.fc1(data))
    
group_info = {}
for key, value in group_dict.items():
    # 숫자 부분을 추출하여 새로운 리스트에 저장
    number_list = []
    for string in value:
        number = 12 + int(string.split('_')[-1][1:])  # '_'를 기준으로 분할하고, 마지막 요소를 선택하여 숫자 부분 추출
        number_list.append(number)
    group_info[key] = (value, number_list, len(number_list))       # 실제 column명, data의 index, 몇 개의 food feature 합쳐서 group 만들었는지.

# Exclusive Grouping
# Exclusive Grouping
class GroupingExclusiveMLP1(nn.Module):
    def __init__(self, basic_dim, hidden_dim, output_size, group_info):
        """
        group_info : {group_name : (food feature, len(food_feature))}
        """
        super(GroupingExclusiveMLP1, self).__init__()
        # group_real_info[key] = (value, number_list, len(number_list))       # 실제 column명, data의 index, 몇 개의 food feature 합쳐서 group 만들었는지.
        self.group_info = list(group_info.values())
        
        self.grouping_layer = nn.ModuleList([])     # 여기서 모든 weight가 뽑히고 21 dim으로 나옴.
        for value in self.group_info:
            self.grouping_layer.append(Linear_layer(value[2], 1))
        
        self.total_fc1 = nn.Linear(21+basic_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.total_fc2 = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
            
        # self.weight_fc = nn.Linear(weight_dim, group_dim)
        # self.relu1 = nn.ReLU()
        # self.total_fc1 = nn.Linear(group_dim + basic_dim, hidden_dim)
        # self.relu2 = nn.ReLU()
        # self.total_fc2 = nn.Linear(hidden_dim, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, data): # basic_x, weight_x):
        # breakpoint()
        grouped_weight = []
        
        g1 = self.grouping_layer[0](data[:, self.group_info[0][1]])
        g2 = self.grouping_layer[1](data[:, self.group_info[1][1]])
        g3 = self.grouping_layer[2](data[:, self.group_info[2][1]])
        g4 = self.grouping_layer[3](data[:, self.group_info[3][1]])
        g5 = self.grouping_layer[4](data[:, self.group_info[4][1]])
        g6 = self.grouping_layer[5](data[:, self.group_info[5][1]])
        g7 = self.grouping_layer[6](data[:, self.group_info[6][1]])
        g8 = self.grouping_layer[7](data[:, self.group_info[7][1]])
        g9 = self.grouping_layer[8](data[:, self.group_info[8][1]])
        g10 = self.grouping_layer[9](data[:, self.group_info[9][1]])
        
        g11 = self.grouping_layer[10](data[:, self.group_info[10][1]])
        g12 = self.grouping_layer[11](data[:, self.group_info[11][1]])
        g13 = self.grouping_layer[12](data[:, self.group_info[12][1]])
        g14 = self.grouping_layer[13](data[:,self.group_info[13][1]])
        g15 = self.grouping_layer[14](data[:,self.group_info[14][1]])
        g16 = self.grouping_layer[15](data[:, self.group_info[15][1]])
        g17 = self.grouping_layer[16](data[:, self.group_info[16][1]])
        g18 = self.grouping_layer[17](data[:, self.group_info[17][1]])
        g19 = self.grouping_layer[18](data[:, self.group_info[18][1]])
        g20 = self.grouping_layer[19](data[:, self.group_info[19][1]])
        
        g21 = self.grouping_layer[20](data[:, self.group_info[20][1]])
        
        grouped_weight = torch.stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20, g21], dim=1).squeeze(-1)       # squeeze(-1)
        
        mixed_ftr = torch.cat((data[:, :13], grouped_weight), dim=1)
        x = self.relu1(self.total_fc1(mixed_ftr))
        x = self.sigmoid(self.total_fc2(x))
        return x
    

# 임상적인 그룹을 그냥 더하는게 아니고 linear combination으로 만듦.
class GroupingExclusiveMLP(nn.Module):
    def __init__(self, basic_dim, hidden_dim, output_size, group_info):
        """
        group_info : {group_name : (food feature, len(food_feature))}
        """
        super(GroupingExclusiveMLP, self).__init__()
        # group_real_info[key] = (value, number_list, len(number_list))       # 실제 column명, data의 index, 몇 개의 food feature 합쳐서 group 만들었는지.
        self.group_info = list(group_info.values())
        
        
        self.group1  = Linear_layer(3, 1)
        self.group2  = Linear_layer(5, 1)
        self.group3  = Linear_layer(5, 1)
        self.group4  = Linear_layer(7, 1)
        self.group5  = Linear_layer(4, 1)
        self.group6  = Linear_layer(2, 1)
        self.group7  = Linear_layer(1, 1)
        self.group8  = Linear_layer(3, 1)
        self.group9  = Linear_layer(1, 1)
        self.group10 = Linear_layer(18, 1)
        
        self.group11 = Linear_layer(5, 1)
        self.group12 = Linear_layer(2, 1)
        self.group13 = Linear_layer(12, 1)
        self.group14 = Linear_layer(9, 1)
        self.group15 = Linear_layer(1, 1)
        self.group16 = Linear_layer(1, 1)
        self.group17 = Linear_layer(15, 1)
        self.group18 = Linear_layer(2, 1)
        self.group19 = Linear_layer(5, 1)
        self.group20 = Linear_layer(2, 1)
        
        self.group21 = Linear_layer(2, 1)
        
        
        self.total_fc1 = nn.Linear(21+basic_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.total_fc2 = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
            
        # self.weight_fc = nn.Linear(weight_dim, group_dim)
        # self.relu1 = nn.ReLU()
        # self.total_fc1 = nn.Linear(group_dim + basic_dim, hidden_dim)
        # self.relu2 = nn.ReLU()
        # self.total_fc2 = nn.Linear(hidden_dim, output_size)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, data): # basic_x, weight_x):
        # breakpoint()
        grouped_weight = []
        
        g1 = self.group1(data[:, self.group_info[0][1]])
        g2 = self.group2(data[:, self.group_info[1][1]])
        g3 = self.group3(data[:, self.group_info[2][1]])
        g4 = self.group4(data[:, self.group_info[3][1]])
        g5 = self.group5(data[:, self.group_info[4][1]])
        g6 = self.group6(data[:, self.group_info[5][1]])
        g7 = self.group7(data[:, self.group_info[6][1]])
        g8 = self.group8(data[:, self.group_info[7][1]])
        g9 = self.group9(data[:, self.group_info[8][1]])
        g10 =self.group10(data[:, self.group_info[9][1]])
        
        g11 = self.group11(data[:, self.group_info[10][1]])
        g12 = self.group12(data[:, self.group_info[11][1]])
        g13 = self.group13(data[:, self.group_info[12][1]])
        g14 = self.group14(data[:, self.group_info[13][1]])
        g15 = self.group15(data[:, self.group_info[14][1]])
        g16 = self.group16(data[:, self.group_info[15][1]])
        g17 = self.group17(data[:, self.group_info[16][1]])
        g18 = self.group18(data[:, self.group_info[17][1]])
        g19 = self.group19(data[:, self.group_info[18][1]])
        g20 = self.group20(data[:, self.group_info[19][1]])
        
        g21 = self.group21(data[:, self.group_info[20][1]])
        
        grouped_weight = torch.cat((g1, g2), dim=1)
        grouped_weight = torch.cat((grouped_weight, g3), dim=1)
        grouped_weight = torch.cat((grouped_weight, g4), dim=1)
        grouped_weight = torch.cat((grouped_weight, g5), dim=1)
        grouped_weight = torch.cat((grouped_weight, g6), dim=1)
        grouped_weight = torch.cat((grouped_weight, g7), dim=1)
        grouped_weight = torch.cat((grouped_weight, g8), dim=1)
        grouped_weight = torch.cat((grouped_weight, g9), dim=1)
        grouped_weight = torch.cat((grouped_weight, g10), dim=1)
        
        grouped_weight = torch.cat((grouped_weight, g11), dim=1)
        grouped_weight = torch.cat((grouped_weight, g12), dim=1)
        grouped_weight = torch.cat((grouped_weight, g13), dim=1)
        grouped_weight = torch.cat((grouped_weight, g14), dim=1)
        grouped_weight = torch.cat((grouped_weight, g15), dim=1)
        grouped_weight = torch.cat((grouped_weight, g16), dim=1)
        grouped_weight = torch.cat((grouped_weight, g17), dim=1)
        grouped_weight = torch.cat((grouped_weight, g18), dim=1)
        grouped_weight = torch.cat((grouped_weight, g19), dim=1)
        grouped_weight = torch.cat((grouped_weight, g20), dim=1)
        
        grouped_weight = torch.cat((grouped_weight, g21), dim=1)

        
        # grouped_weight = torch.stack([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20, g21], dim=1).squeeze(-1)       # squeeze(-1)
        
        mixed_ftr = torch.cat((data[:, :13], grouped_weight), dim=1)
        x = self.relu1(self.total_fc1(mixed_ftr))
        x = self.sigmoid(self.total_fc2(x))
        return x

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 초기 hidden state 설정
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 마지막 시간 단계의 hidden state를 사용하여 출력
        out = self.sigmoid(out)  # 시그모이드 함수를 통해 이진 분류를 위한 확률값으로 변환
        return out
    
    
class DurationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(DurationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_duration = nn.Linear(hidden_size, output_size)
        self.fc_onset = nn.Linear(hidden_size, output_size)
        self.sigmoid_duration = nn.Sigmoid()
        self.sigmoid_onset = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 초기 hidden state 설정
        out, _ = self.rnn(x, h0)
        
        # out[:,0,:]은 부산대 할일 6번 그림의 h_1이고, out[:,-1,:]은 부산대 할일 6번 그림의 h_2이다.
        out_onset, out_duration = self.fc_onset(out[:, 0, :]), self.fc_duration(out[:, -1, :])
        out_duration = self.sigmoid_duration(out_duration)  # 시그모이드 함수를 통해 이진 분류를 위한 확률값으로 변환
        out_onset = self.sigmoid_onset(out_onset)  # 시그모이드 함수를 통해 이진 분류를 위한 확률값으로 변환
        return out_onset, out_duration