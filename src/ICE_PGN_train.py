from network import ResNet18_PGN
import numpy as np
import torch
from torch import nn, optim
import argparse
from gensim.models import LdaModel
from lda_topic_model import gen_corpus
import time
from slc_dataset import Comp_Ice_Dataset_un
import transform_data
from torch.utils.data import DataLoader
from torchvision import transforms
import joblib
import warnings
import os
from tensorboardX import SummaryWriter
from lda_topic_model import transform_gensim_corpus, lda_test_doc
from loss import SoftConstraintLoss


warnings.simplefilter(action='ignore', category=FutureWarning)

SCAT_CLASS = 9


def parameter_setting(args):
    config = {}
    config['topic_num'] = args.topic_num
    config['lda_model'] = args.lda_model
    config['kmeans_model'] = args.kmeans_model
    config['datatxt_train'] = args.datatxt_train
    config['batch_size'] = args.batch_size
    config['device'] = args.device
    config['num_epochs'] = args.num_epochs
    config['data_root'] = args.data_root
    config['models'] = {'pretrained': args.pretrained,
                        'save_model_path': args.save_model_path}
    config['non_zero_prob'] = args.non_zero_prob
    config['iscontinue'] = args.iscontinue

    return config


def load_pretrained_model(config):
    topic_num = config['topic_num']

    if config['iscontinue']:
        model = ResNet18_PGN(topic_num)
        model.load_state_dict(torch.load(config['models']['pretrained']))
        print('continue training')
    else:
        net = torch.load(config['models']['pretrained'])
        model = ResNet18_PGN(topic_num)
        # model.load_state_dict(net)
        for name in net.keys():
            if 'fc' not in name and 'layer4' not in name:
                model.state_dict()[name].copy_(net[name])
        print('train CompNet')

    return model


def get_dataloader(config):
    data_transforms = transforms.Compose([
        transform_data.Normalize_img(mean=0.437279412021903, std=0.03280072512541247),  # ICE
        transform_data.Numpy2Tensor_img(channels=3)
    ])
    dataset_train = Comp_Ice_Dataset_un(data_txt=config['datatxt_train'],
                                        data_root=config['data_root'],
                                        transform=data_transforms)


    dataloader = {}
    dataloader['train'] = DataLoader(dataset_train,
                                     batch_size=config['batch_size'][0],
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=0,
                                     pin_memory=True)

    return dataloader


def get_BoT(scat_paths, kmeans, lda, num_topics):
    """
    generate the topic combinition
    :param scat: torch.Tensor, batch_size * patch_size * patch_size
    :param config:
    :return:
    """
    word_win = 8
    stride = word_win
    bot = []

    for scat_path in scat_paths:
        scat_patch = np.load(scat_path)
        test_docs = scat_patch.reshape([1, scat_patch.shape[0], scat_patch.shape[1]])
        test_corpus = gen_corpus(test_docs, kmeans, word_win, stride)
        test_gensim_corpus = transform_gensim_corpus(test_corpus)
        topic = lda_test_doc(test_gensim_corpus[0], lda)

        attr = np.zeros(num_topics)
        for tt in topic:
            attr[tt[0]] = tt[1]
        bot.append(attr)

    return np.array(bot)


def train(config):
    global SCAT_CLASS

    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())

    model = ResNet18_PGN(config['topic_num'])  # retrain
    # model = load_pretrained_model(config) # continue training
    lda_model = LdaModel.load(config['lda_model'])
    kmeans_model = joblib.load(config['kmeans_model'])

    nonzero_prob = config['non_zero_prob']

    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    writer = SummaryWriter('../log/' + config['models']['save_model_path'].split('/')[-1] + 'log')

    model.to(device)
    SoftConstraint = SoftConstraintLoss(nonzero_prob)
    dataloader = get_dataloader(config)

    for epoch in range(config['num_epochs']):
        # for each epoch, load new data
        timer = time.perf_counter()
        for sample in dataloader['train']:
            model.train()
            scat_paths = sample['scat_path']
            data = sample['img']
            attr = get_BoT(scat_paths, kmeans_model, lda_model, config['topic_num']) # shape: batchsize * topic_num
            feat = model(data.to(device))

            loss = SoftConstraint(feat, torch.FloatTensor(attr).to(device))
            print('loss_comp: ' + str(loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)

        print('epoch ' + str(epoch + 1) + ':\tloss', loss.item())
        print(str(time.perf_counter() - timer))
        writer.add_scalar('loss', loss.item(), epoch + 1)

        torch.save(model.state_dict(), config['models']['save_model_path'] + 'epoch' + str(epoch+1) + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='PGN train')

    parser.add_argument('--pretrained', default='..')
    parser.add_argument('--lda_model', default='../result/ICE_lda_175.pkl')
    parser.add_argument('--kmeans_model', default='../result/ICE_kmeans.pkl')
    parser.add_argument('--data_root', default='../data/SeaIceData/')
    parser.add_argument('--datatxt_train', default='../data/ICE_dataset.txt')
    parser.add_argument('--topic_num', type=int, default=175)
    parser.add_argument('--batch_size', type=int, nargs='+', default=[300, 32])
    parser.add_argument('--device', default='0')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--non_zero_prob', type=float, default=0.9)
    parser.add_argument('--save_model_path', default='../model/ICE_PGN_soft_topic175_retrain_')
    parser.add_argument('--iscontinue', type=int, default=0)

    args = parser.parse_args()
    config = parameter_setting(args)

    train(config)

