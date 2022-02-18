from network import ResNet18_PGIL, ResNet18_PGN
import torch
from torch import nn, optim
import argparse
from slc_dataset import Comp_Ice_Dataset
import transform_data
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings
import os
from tensorboardX import SummaryWriter
from learning_schedule import net_param_setting


warnings.simplefilter(action='ignore', category=FutureWarning)


def parameter_setting(args):
    config = {}
    config['data_root'] = args.data_root
    config['datatxt_train'] = args.datatxt_train
    config['datatxt_valid'] = args.datatxt_valid
    config['cate_txt'] = args.cate_txt
    config['cate_num'] = args.cate_num

    config['batch_size'] = args.batch_size
    config['device'] = args.device
    config['num_epochs'] = args.num_epochs
    config['models'] = {'pretrained': args.pretrained,
                        'save_model_path': args.save_model_path}

    config['topic_num'] = args.topic_num
    config['resume_from'] = args.resume_from
    return config



def get_dataloader(config):

    data_transforms = transforms.Compose([
        transform_data.Normalize_img(mean=0.437279412021903, std=0.03280072512541247), # ICE
        transform_data.Numpy2Tensor_img(channels=3)
    ])
    dataset_train = Comp_Ice_Dataset(data_txt=config['datatxt_train'], cate_txt=config['cate_txt'], data_root=config['data_root'],
                                transform=data_transforms)

    dataset_valid = Comp_Ice_Dataset(data_txt=config['datatxt_valid'], cate_txt=config['cate_txt'], data_root=config['data_root'],
                                transform=data_transforms)
    dataloader = {}
    dataloader['train'] = DataLoader(dataset_train,
                                     batch_size=config['batch_size'][0],
                                     shuffle=True,
                                     num_workers=16,
                                     pin_memory=True)
    dataloader['valid'] = DataLoader(dataset_valid,
                                     batch_size=config['batch_size'][1],
                                     shuffle=True,
                                     drop_last=False,
                                     num_workers=16,
                                     pin_memory=True)


    train_count_dict = {}
    cate_num = config['cate_num']
    for i in range(cate_num):
        cate_name = dataset_train.cate.loc[dataset_train.cate['label'] == i]['catename'].iloc[0]
        train_count_dict[i] = len(dataset_train.data.loc[dataset_train.data['catename'] == cate_name])
    loss_weight = [
        (1.0 - float(train_count_dict[i]) / float(sum(train_count_dict.values()))) * cate_num / (cate_num - 1)
        for i in range(cate_num)]

    return dataloader, loss_weight



def train(config):
    # prepare the path of log
    save_dir = config['models']['save_model_path']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    log_save_path = os.path.join(save_dir,config['models']['save_model_path'].split('/')[-1] + '_log.txt')

    # gpu setting
    f = open(log_save_path,'a')
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(),config['device'])
    print(torch.cuda.get_device_name(),config['device'],file = f)
    f.close()

    # build PIL network
    model_CNN = ResNet18_PGIL(config['cate_num'])
    model_CNN.to(device)

    net = torch.load('../model/resnet18_tsx.pth')
    for name in net.keys():
        if name[:2] != 'fc':
            model_CNN.state_dict()[name].copy_(net[name])

    # build PGL network
    model_CompNet = ResNet18_PGN(config['topic_num'])
    model_CompNet.load_state_dict(torch.load(config['models']['pretrained']))
    model_CompNet.to(device)
    model_CompNet.eval() 

    # optimizer
    init_lr = 1e-3
    # param_list = net_param_setting(model_CNN, init_lr)
    optimizer = optim.SGD(model_CNN.parameters(), lr=init_lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs']-3, eta_min=1e-6)

    # data preparation
    dataloader, loss_weight = get_dataloader(config)

    # loss setting
    loss_func = nn.CrossEntropyLoss()


    # tensorboard log
    writer = SummaryWriter(os.path.join(save_dir,config['models']['save_model_path'].split('/')[-1] + '_log'))

    for epoch in range(config['num_epochs']):
        model_CNN.train()
        for sample in dataloader['train']:
            data = sample['img']
            labels = sample['label'].to(device)
            comp_feat = model_CompNet(data.to(device))
            output = model_CNN(data.to(device), comp_feat)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            del labels, output

        if epoch < config['num_epochs']-3:  # update lr
            scheduler.step()

        # test after each epoch training
        acc_num = 0.0
        data_num = 0
        val_loss_ce = 0.0
        count = 0

        model_CNN.eval()
        with torch.no_grad():
            for sample in dataloader['valid']:
                data = sample['img']
                labels = sample['label'].to(device)
                comp_feat = model_CompNet(data.to(device))
                output = model_CNN(data.to(device), comp_feat)
                _, pred = torch.Tensor.max(output, 1)
                acc_num += torch.sum(torch.squeeze(pred) == labels.data).float()
                data_num += labels.size()[0]
                val_loss_ce += loss_func(output, labels).item()
                count += 1
                del labels, output, pred

        val_loss_ce /= count
        val_acc = acc_num / data_num
        print('---- epoch ' + str(epoch + 1) + '\tloss_ce ' + str(loss.item()) + '----------------------------- lr: ' + str(optimizer.param_groups[0]['lr']))
        print('---- test acc: ' + str(val_acc.item()) + '\tval_loss_ce ' + str(val_loss_ce))
        f = open(log_save_path,'a')
        print('---- epoch ' + str(epoch + 1) + '\tloss_ce ' + str(loss.item()) + '----------------------------- lr: ' + str(optimizer.param_groups[0]['lr']),file = f)
        print('---- test acc: ' + str(val_acc.item()) + '\tval_loss_ce ' + str(val_loss_ce),file = f)
        f.close()
        writer.add_scalars('loss', {'cls_train': loss.item(),  
                                    'cls_val': val_loss_ce,},
                           epoch+1)
        writer.add_scalar('val_acc', val_acc, epoch+1)
        # torch.save(model_CNN.state_dict(), os.path.join(save_dir,config['models']['save_model_path'].split('/')[-1] + '_epoch' + str(epoch + 1) + '.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='CNN_training')

    # PGL setting
    parser.add_argument('--topic_num', type=int, default=175)
    parser.add_argument('--pretrained', default='../model/ICE_PGN_soft_topic175_retrain.pth')
    parser.add_argument('--resume_from', default=None)

    # dataset setting
    parser.add_argument('--data_root', default='../data/SeaIceData/')
    parser.add_argument('--datatxt_train', default='../data/1_ICE_dataset_c7_train45.txt')
    parser.add_argument('--datatxt_valid', default='../data/ICE_dataset_test_n50_c7.txt')
    parser.add_argument('--cate_txt', default='../data/ICE_catename2label_7.txt')
    parser.add_argument('--cate_num', type=int, default=7)

    # training setting
    parser.add_argument('--batch_size', type=int, nargs='+', default=[200, 200])
    parser.add_argument('--device', default='0')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--save_model_path', default='../model/ICE_PGIL_')

    args = parser.parse_args()
    config = parameter_setting(args)

    train(config)




