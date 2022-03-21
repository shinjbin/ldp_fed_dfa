import random
import time
import threading
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process
from torch.utils.data import DataLoader

from src.dataset import Dataset
from src.aggregation import Aggregation


# def run(rank, size):
#     tensor = torch.zeros(1)
#     req = None
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         req = dist.isend(tensor=tensor, dst=1)
#         print('Rank 0 started sending')
#     else:
#         # Receive tensor from process 0
#         req = dist.irecv(tensor=tensor, src=0)
#         print('Rank 1 started receiving')
#     req.wait()
#     print('Rank ', rank, ' has data ', tensor[0])
#
#
# def init_process(rank, size, fn, backend='gloo'):
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)


def k_selection(parameters_list, k):
    return random.choices(parameters_list, k=k)


if __name__ == '__main__':
    start_time = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # hyperparameter
    num_clients = 5
    batch_size = 4
    num_round = 5
    train_mode = 'dfa'  # 'dfa' or 'backprop'
    learning_rate = 0.001
    tol = 0.0005
    ldp = True

    # load datasets and split
    dataset = Dataset()
    train_dataset_split, test_dataset = dataset.split(num_clients)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    path_list = []
    for i in range(num_clients):
        path_list.append(f'CLIENT{i}_MNIST_CLASSIFIER.pth')
    global_path = 'GLOBAL_MNIST_CLASSIFIER.pth'

    # create aggregation module
    aggregation = Aggregation(device, global_path, train_mode)

    # create clients
    clients = []
    print(f'Creating {num_clients} clients...')
    for i in range(num_clients):
        aggregation.create_clients(i)
        aggregation.clients[i].dataload(train_dataset=train_dataset_split[i],
                                        test_dataset=test_dataset,
                                        batch_size=batch_size,
                                        path=path_list[i],
                                        train_mode=train_mode,
                                        lr=learning_rate)

    # federated learning start
    for r in range(num_round):
        print(f'---------------<Round {r}>----------------')

        # train clients local model
        aggregation.train_client(train_mode=train_mode, tol=tol)
        print("--------------------------------")

        # test clients local model
        aggregation.test_client()

        # update global model parameter(parameter averaging)
        aggregation.global_parameter_update(ldp=ldp)

        # test global model
        global_accuracy = aggregation.global_model.test(test_loader)
        print(f'Global model accuracy: {global_accuracy*100:.2f}%')
        print("--------------------------------")

        # local model parameter update
        aggregation.local_parameter_update()

    global_accuracy = aggregation.global_model.test(test_loader)
    aggregation.global_model.example(test_loader)

    # write result file
    f = open("result.txt", "a")
    f.write(f'--------------------------------------<{train_mode}>---------------------------------------------\n'
            f'num_clients = {num_clients} batch_size = {batch_size}, num_round = {num_round}, '
            f'learning_rate = {learning_rate}, tol = {tol}, ldp = {ldp}\n\n'
            f'Final global model accuracy using {train_mode}: {global_accuracy*100:.2f}%\n'
            f'time taken using {train_mode}: {time.time()-start_time:.2f}\n'
            f'---------------------------------------------------------------------------------------------\n\n')
    f.close()

