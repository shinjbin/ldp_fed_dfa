import torch
import copy

from .models import OneHiddenNNModel, OneHiddenNN
from .client import Client


class Aggregation(object):
    def __init__(self, device, path, train_mode):
        self.clients = []
        self.num_client = 0
        self.global_model = OneHiddenNNModel(device=device, path=path, lr=0.001, train_mode=train_mode)

    def create_clients(self, client_id):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        client = copy.deepcopy(Client(client_id=client_id, device=device))
        self.clients.append(client)
        self.num_client += 1

    def train_client(self, train_mode, tol, train=True):
        B1 = torch.randn(10, 800).to('cuda')

        for i in range(self.num_client):
            print(f'Training client {i}...')
            self.clients[i].local_train(train=train, train_mode=train_mode, B1=B1, tol=tol)

    def test_client(self):
        for i in range(self.num_client):
            print(f"Client {i} testing...")
            self.clients[i].local_eval()

    def parameter_client(self, index):
        W1, W2, b1, b2 = self.clients[index].nn.model.get_parameters()
        return W1, W2, b1, b2

    def average_parameters(self, ldp=True):
        avg_W1, avg_W2, avg_b1, avg_b2 = 0, 0, 0, 0
        for i in range(self.num_client):
            if ldp:
                ldp_W1, ldp_W2, ldp_b1, ldp_b2 = self.clients[i].ldp(alpha=0.1, c=1, rho=3)

                avg_W1 += ldp_W1
                avg_W2 += ldp_W2
                avg_b1 += ldp_b1
                avg_b2 += ldp_b2

            else:
                temp_W1, temp_W2, temp_b1, temp_b2 = self.clients[i].nn.model.get_parameters()
                temp_W1, temp_W2, temp_b1, temp_b2 = temp_W1.to('cuda'), temp_W2.to('cuda'), temp_b1.to('cuda'), temp_b2.to('cuda')
                avg_W1 += temp_W1
                avg_W2 += temp_W2
                avg_b1 += temp_b1
                avg_b2 += temp_b2

            # check_w1 = torch.equal(ldp_W1, temp_W1)
            # check_w2 = torch.equal(ldp_W1, temp_W1)
            # check_b1 = torch.equal(ldp_W1, temp_W1)
            # check_b2 = torch.equal(ldp_W1, temp_W1)
            # print(check_w1, check_w2, check_b1, check_b2)

        avg_W1 /= self.num_client
        avg_W2 /= self.num_client
        avg_b1 /= self.num_client
        avg_b2 /= self.num_client

        return avg_W1, avg_W2, avg_b1, avg_b2

    def global_parameter_update(self, ldp=True):
        print("--------------------------------")
        print('parameter averaging...')

        W1, W2, b1, b2 = self.average_parameters(ldp=ldp)

        self.global_model.model.parameter_renew(W1, W2, b1, b2)

    def local_parameter_update(self):
        print('local model parameter updating...')
        W1, W2, b1, b2 = self.global_model.model.get_parameters()

        for i in range(self.num_client):
            self.clients[i].nn.model.parameter_renew(W1, W2, b1, b2)

        print("--------------------------------")



