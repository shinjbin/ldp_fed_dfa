import numpy as np
import random
import torch
import copy


class LDP(object):
    def __init__(self, W1_user, W2_user, b1_user, b2_user, alpha, c, rho):
        self.W1 = copy.deepcopy(W1_user).to('cuda') * (10**rho)
        self.W2 = copy.deepcopy(W2_user).to('cuda') * (10**rho)
        self.b1 = copy.deepcopy(b1_user).to('cuda') * (10**rho)
        self.b2 = copy.deepcopy(b2_user).to('cuda') * (10**rho)
        self.a = alpha
        self.v = None
        self.score = {}
        self.u = []
        for i in range(-c*(10**rho), c*(10**rho)+1):
            self.u.append(i)
        self.u = torch.tensor(self.u).to('cuda')

    def ordinal_cldp(self):
        print("ordinal CLDP running...")
        for i in range(self.W1.shape[0]):
            for j in range(self.W1.shape[1]):

                d = torch.abs(torch.sub(self.u, self.W1[i][j]))
                score = torch.exp(-self.a * d / 2)
                # prob = []
                # sum_sc = torch.sum(score).to('cuda')
                # for s in score:
                #     prob.append(s / sum_sc)
                # prob = torch.tensor(prob)

                idx = score.multinomial(num_samples=1, replacement=True)
                with torch.no_grad():
                    self.W1[i][j] = self.u[idx]

        for i in range(self.W2.shape[0]):
            for j in range(self.W2.shape[1]):
                d = torch.abs(torch.sub(self.u, self.W2[i][j]))
                score = torch.exp(-self.a * d / 2)
                # prob = []
                # sum_sc = torch.sum(score)
                # for s in score:
                #     prob.append(s / sum_sc)
                # prob = torch.tensor(prob)

                idx = score.multinomial(num_samples=1, replacement=True)
                with torch.no_grad():
                    self.W2[i][j] = self.u[idx]

        for i in range(self.b1.shape[0]):
            d = torch.abs(torch.sub(self.u, self.b1[i]))
            score = torch.exp(-self.a * d / 2)
            # prob = []
            # sum_sc = torch.sum(score)
            # for s in score:
            #     prob.append(s / sum_sc)
            # prob = torch.tensor(prob)

            idx = score.multinomial(num_samples=1, replacement=True)
            with torch.no_grad():
                self.b1[i] = self.u[idx]

        for i in range(self.b2.shape[0]):
            d = torch.abs(torch.sub(self.u, self.b2[i]))
            score = torch.exp(-self.a * d / 2)
            # prob = []
            # sum_sc = torch.sum(score)
            # for s in score:
            #     prob.append(s / sum_sc)
            # prob = torch.tensor(prob)

            idx = score.multinomial(num_samples=1, replacement=True)
            with torch.no_grad():
                self.b2[i] = self.u[idx]

        self.W1 /= 1000
        self.W2 /= 1000
        self.b1 /= 1000
        self.b2 /= 1000

        return self.W1, self.W2, self.b1, self.b2
