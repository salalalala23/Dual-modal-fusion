import torch
import torch.nn as nn
from indicators.kappa import *
from tqdm import tqdm


def test(model, test_loader, mode='2'):
    model.eval()
    test_loss = 0
    test_matrix = np.zeros([len(colormap), len(colormap)])
    loop = tqdm(test_loader, leave=True)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        if mode == '1':
            for batch_idx, (data, target, _, _) in enumerate(loop):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data, mode=mode)
                test_loss += criterion(output, target.long())
                pred = output.data.max(1, keepdim=True)[1]
                for i in range(len(target)):
                    test_matrix[int(pred[i].item())][int(target[i].item())] += 1
                loop.set_postfix(mode='test')
            loop.close()
        elif mode == '2':
            for batch_idx, (data1, data2, target, _, _) in enumerate(loop):
                data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
                output = model(data1, data2)
                test_loss += criterion(output, target.long())
                pred = output.data.max(1, keepdim=True)[1]
                for i in range(len(target)):
                    test_matrix[int(pred[i].item())][int(target[i].item())] += 1
                loop.set_postfix(mode='test')
            loop.close()
        elif mode == '3':
            for batch_idx, (data1, data2, data3, target, _, _) in enumerate(loop):
                data1, data2, data3, target = data1.to(DEVICE), data2.to(DEVICE), data3.to(DEVICE), target.to(DEVICE)
                output = model(data1, data2, data3)
                test_loss += criterion(output, target.long())
                pred = output.data.max(1, keepdim=True)[1]
                for i in range(len(target)):
                    test_matrix[int(pred[i].item())][int(target[i].item())] += 1
                loop.set_postfix(mode='test')
            loop.close()
    return test_matrix


def test_second(model, test_loader, inpo, mode='2'):
    model.load_state_dict(torch.load(inpo))
    model.eval()
    test_loss = 0
    test_matrix = np.zeros([len(colormap), len(colormap)])
    loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for batch_idx, (data1, data2, target, _, _) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            output = model(data1, data2)
            # test_loss += nn.CrossEntropyLoss(output, target.long())
            pred = output.data.max(1, keepdim=True)[1]
            for i in range(len(target)):
                test_matrix[int(pred[i].item())][int(target[i].item())] += 1
            loop.set_postfix(mode='test')
    loop.close()
    return test_matrix


def test_CT(model, test_loader, mode='2'):
    model.eval()
    test_loss = 0
    test_matrix = np.zeros([len(colormap), len(colormap)])
    loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            output, _ = model(data1, data2)
            # test_loss += nn.CrossEntropyLoss(output, target.long())
            pred = output.data.max(1, keepdim=True)[1]
            for i in range(len(target)):
                test_matrix[int(pred[i].item())][int(target[i].item())] += 1
            loop.set_postfix(mode='test')
        loop.close()
    return test_matrix


def test_SCPF(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_matrix = np.zeros([len(colormap), len(colormap)])
    loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, target, _, _) in enumerate(loop):
            data1, data2, data3, target = data1.to(DEVICE), data2.to(DEVICE), data3.to(DEVICE), target.to(DEVICE)
            output = model(data1, data2, data3)
            test_loss += criterion(output, target.long())
            pred = output.data.max(1, keepdim=True)[1]
            for i in range(len(target)):
                test_matrix[int(pred[i].item())][int(target[i].item())] += 1
            loop.set_postfix(mode='test')
    loop.close()
    np.save(data_city+"_result.npy", test_matrix)
    return test_matrix


def test_SCLN(model, test_loader, triplet_loss):
    model.eval()
    test_loss = 0
    # test_matrix = []
    loop = tqdm(test_loader, leave=True)
    # for i in range(4):
    test_matrix = np.zeros([len(colormap), len(colormap)])
    with torch.no_grad():
        for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(loop):
            data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
            data3, data4 = data3.to(DEVICE), data4.to(DEVICE)
            bs = len(data1)
            data = torch.concat([data1, data2, data3, data4])
            out = model(data)
            target1 = torch.cat((target, target))
            target2 = torch.cat((target, target, target, target))
            pred1 = (out[:bs]+out[bs:2*bs]).softmax(dim=-1).data.max(1, keepdim=True)[1]
            # pred2 = out2.data.max(1, keepdim=True)[1]
            # pred3 = out3.data.max(1, keepdim=True)[1]
            # pred4 = out4.data.max(1, keepdim=True)[1]
            for i in range(len(target)):
                test_matrix[int(pred1[i].item())][int(target[i].item())] += 1
            #     test_matrix[1][int(pred2[i].item())][int(target[i].item())] += 1
            # for i in range(len(target1)):
            #     test_matrix[2][int(pred3[i].item())][int(target1[i].item())] += 0.5
            # for i in range(len(target2)):
            #     test_matrix[3][int(pred4[i].item())][int(target2[i].item())] += 0.25
            # if batch_idx % int(len(test_loader) / 10) == 0:
            #     print("Test Rate: {:.0f}%".format(100. * batch_idx / len(test_loader)))
            loop.set_postfix(mode='test')
    # model.eval()
    # test_loss = 0
    # test_matrix = np.zeros([len(colormap), len(colormap)])
    # loop = tqdm(test_loader, leave=True)
    # with torch.no_grad():
    #     for batch_idx, (data1, data2, data3, data4, target, _, _) in enumerate(loop):
    #         data1, data2, target = data1.to(DEVICE), data2.to(DEVICE), target.to(DEVICE)
    #         data3, data4 = data3.to(DEVICE), data4.to(DEVICE)
    #         bs = len(data1)
    #         data_origin = torch.cat([data1, data2], dim=0)
    #         data_generate = torch.cat([data3, data4], dim=0)
    #         target1 = torch.cat((target, target))
    #         target2 = torch.cat((target, target, target, target))
    #         out = model(data_origin, data_generate)
    #         test_loss += criterion(out[:2*bs], target1.long())
    #         pred = out[:2*bs].data.max(1, keepdim=True)[1]
    #         for i in range(len(target1)):
    #             test_matrix[int(pred[i].item())][int(target1[i].item())] += 0.5
    #         # if batch_idx % int(len(test_loader) / 10) == 0:
    #         #     print("Test Rate: {:.0f}%".format(100. * batch_idx / len(test_loader)))
    #         loop.set_postfix(mode='test')
    loop.close()
    np.save(data_city + "_result.npy", test_matrix)
    # for i in range(4):
    #     # test_matrix[i] = test_matrix[i] / 2
    #     aa, oa, correct, k = aa_oa(test_matrix[i])
    #     test_loss /= len(test_loader.dataset)
    #     print("\nTest set: Average loss:{:.4f}. Accuracy: {}/{} ({:.6f}%) Kappa:{:.6f} AA:{:.6f} OA:{:.6f}\n"
    #         .format(test_loss, correct, len(test_loader.dataset),
    #         100. * correct / len(test_loader.dataset), k, aa, oa))
    return test_matrix
