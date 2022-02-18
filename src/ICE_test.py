from network import ResNet18_PGN, ResNet18_PGIL
import numpy as np
import pandas as pd
import torch, argparse, transform_data, warnings, os
from slc_dataset import Comp_Ice_Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Range1d
from bokeh.layouts import row, gridplot
from bokeh.palettes import Pastel1
from bokeh.transform import cumsum
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics
from unsup_comp_attr_ICE import get_BoT
from gensim.models import LdaModel
from sklearn.cluster import KMeans
import joblib
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as func
from sklearn.metrics import precision_score,recall_score
import torch.nn.functional as func
from scipy.interpolate import interp1d


warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(suppress=True)

def parameter_setting(args):
    config = {}
    config['datatxt_valid'] = args.datatxt_valid
    config['cate_txt'] = args.cate_txt
    config['batch_size'] = args.batch_size
    config['device'] = args.device
    config['data_root'] = args.data_root
    config['models'] = {'pretrained': args.pretrained}
    config['cate_num'] = args.cate_num
    config['net_type'] = args.net_type
    config['topic_num'] = args.topic_num

    return config

def load_pretrained_model(config):

    if config['net_type'] == 'PGN':
        model = ResNet18_PGN(config['topic_num'])
    elif config['net_type'] == 'PGIL':
        model = ResNet18_PGIL(config['cate_num'])
    else:
        Exception('please input correct model config')
    model.load_state_dict(torch.load(config['models']['pretrained'], map_location=torch.device('cpu')))
    return model

def get_dataloader(config):
    data_transforms = transforms.Compose([
        transform_data.Normalize_img(mean=0.437279412021903, std=0.03280072512541247), # ICE
        transform_data.Numpy2Tensor_img(channels=3)
    ])

    dataset_valid = Comp_Ice_Dataset(data_txt=config['datatxt_valid'],
                                     cate_txt=config['cate_txt'],
                                     data_root=config['data_root'],
                                     transform=data_transforms)

    dataloader = {}

    dataloader['valid'] = DataLoader(dataset_valid,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            drop_last=False,
                            num_workers=0)

    return dataloader

def get_dataloader_SVM(config, train_txt):
    data_transforms = transforms.Compose([
        transform_data.Normalize_img(mean=0.437279412021903, std=0.03280072512541247),  # ICE
        transform_data.Numpy2Tensor_img(channels=3)
    ])

    dataset_train = Comp_Ice_Dataset(data_txt=train_txt, cate_txt=config['cate_txt'],
                                     data_root=config['data_root'],
                                     transform=data_transforms)

    dataset_valid = Comp_Ice_Dataset(data_txt=config['datatxt_valid'], cate_txt=config['cate_txt'],
                                     data_root=config['data_root'],
                                     transform=data_transforms)

    dataloader = {}
    dataloader['train'] = DataLoader(dataset_train,
                                     batch_size=config['batch_size'],
                                     num_workers=0)
    dataloader['valid'] = DataLoader(dataset_valid,
                                     batch_size=config['batch_size'],
                                     drop_last=False,
                                     num_workers=0)

    return dataloader


def matrix_analysis(label_pred, label_true, cate_num):
    """

    :param label_pred: np.array, 1 dim
    :param label_true: np.array, 1 dim
    :param cate_num:
    :return:
    """
    matrix = np.zeros([cate_num, cate_num])

    for i in range(cate_num):
        index = np.where(label_true == i)[0]
        if index.size != 0:
            label_term = label_pred[index]
            for j in range(cate_num):
                matrix[i, j] = len(np.where(label_term == j)[0])

    return matrix


def PGIL_test(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_pretrained_model(config)

    model_CompNet = ResNet18_PGN(config['topic_num'])
    model_CompNet.load_state_dict(torch.load('../model/ICE_PGN_soft_topic175_retrain.pth'))
    model_CompNet.to(device)
    model_CompNet.eval()

    dataloader = get_dataloader(config)

    model.to(device)
    model.eval()
    pred_labels = []
    true_labels = []

    with torch.no_grad():
        for sample in dataloader['valid']:
            data = sample['img']
            labels = sample['label'].to(device)
            comp_feat = model_CompNet(data.to(device))
            output = model(data.to(device), comp_feat) # comp_cnn_2
            # _, output = model(data.to(device))

            _, pred = torch.Tensor.max(output, 1)

            pred_labels = pred_labels + list(np.array(torch.squeeze(pred).data.cpu()))
            true_labels = true_labels + list(np.array(labels.data.cpu()))

            del labels, output, pred

    return np.array(pred_labels), np.array(true_labels)


def topic_sparsity(config):
    config['topic_num'] = 25
    print(config['topic_num'])
    lda_model = LdaModel.load('../result/ICE_lda_25.pkl')
    kmeans_model = joblib.load('../result/ICE_kmeans.pkl')
    dataloader = get_dataloader(config)

    topics_arr = np.zeros([0, config['topic_num']])
    for sample in dataloader['valid']:
        scat_paths = sample['scat_path']
        attr = get_BoT(scat_paths, kmeans_model, lda_model, config['topic_num'])  # shape: batchsize * topic_num
        topics_arr = np.concatenate((topics_arr, attr), axis=0)  # shape: N * 50

    n = config['topic_num']
    # Norm_L1_L2 = np.sum(topics_arr, axis=1) / np.sqrt(np.sum(topics_arr * topics_arr, axis=1))
    # return np.mean((np.sqrt(n) - Norm_L1_L2) / (np.sqrt(n) - 1.0))
    return np.mean(np.sum(topics_arr <= 0.01, axis=1) / n)


def SVM_test(config, train_txt):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_pretrained_model(config)

    dataloader = get_dataloader_SVM(config, train_txt)

    model.to(device)
    model.eval()

    embs_arr = np.zeros([0, 256])
    label_arr = np.zeros([0, ])
    for sample in dataloader['train']:
        data = sample['img']
        label_arr = np.concatenate((label_arr, sample['label'].numpy()), axis=0)  # shape: N,
        embs_arr = np.concatenate((embs_arr, model(data.cuda()).cpu().data.numpy()), axis=0)  # shape: N * 50

    embs_min = np.min(embs_arr)
    embs_max = np.max(embs_arr)

    embs_arr = (embs_arr - embs_min) / (embs_max - embs_min)
    classifier = SVC(kernel='linear', class_weight='balanced', probability=True)
    classifier.fit(embs_arr, label_arr)

    pred_labels = []
    true_labels = []
    test_paths = []

    for sample in dataloader['valid']:
        data = sample['img']
        labels = sample['label'].to(device)
        test_paths += sample['scat_path']

        true_labels = true_labels + list(np.array(labels.data.cpu()))
        embs_valid = model(data.cuda()).cpu().data
        embs_valid = (embs_valid - embs_min) / (embs_max - embs_min)

        pred_labels = pred_labels + list(np.argsort(classifier.decision_function(embs_valid))[:,-1])
        del labels

    # label2name = pd.read_csv(config['cate_txt'])
    # pred_catename = [label2name.loc[pred_labels[i]]['catename'] for i in range(len(pred_labels))]
    # df = pd.DataFrame(columns=['path', 'catename'])
    # df['path'] = pd.Series(test_paths)
    # df['catename'] = pd.Series(pred_catename)
    # df.to_csv('/home/hzl/STAT/pytorch_code/ICE/data/GF3_Paris_compnet_hdec.txt', index=False)
    return np.array(pred_labels), np.array(true_labels)


def get_score_label(df_data, label=None):
    if label == None:
        y_path = list(df_data.pop('path'))
        y_label = np.array(df_data.pop('label'), dtype=int)
        y_score = np.array(df_data)
    else:
        df_data1 = df_data.loc[df_data['label'] == label]
        y_path = list(df_data1.pop('path'))
        y_label = np.array(df_data1.pop('label'), dtype=int)
        y_score = np.array(df_data1)

    return y_score, y_label, y_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ICE_test')
    parser.add_argument('--pretrained', default='../model/ICE_PGN_soft_topic175_retrain.pth') #
    parser.add_argument('--net_type', default='PGN') # PGN, PGIL
    parser.add_argument('--data_root', default='../data/SeaIceData/') # GF3_Paris_annotate
    parser.add_argument('--datatxt_valid', default='../data/ICE_dataset_test_n50_c7.txt') # ICE_dataset_test_n50_c7
    parser.add_argument('--cate_txt', default='../data/ICE_catename2label_7.txt')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', default='0')
    parser.add_argument('--cate_num', type=int, default=7)
    parser.add_argument('--topic_num', type=int, default=175)

    args = parser.parse_args()
    config = parameter_setting(args)
    # model = load_pretrained_model(config)
    print(config['models']['pretrained'])

    # print(topic_sparsity(config))

    """ PGN+SVM Test
    """
    pred_labels, true_labels = SVM_test(config, '../data/1_ICE_dataset_c7_train45.txt')
    matrix = matrix_analysis(pred_labels, true_labels, config['cate_num'])
    catename2label = pd.read_csv(config['cate_txt'])
    # print(catename2label)
    print(matrix)
    print(metrics.classification_report(true_labels, pred_labels, target_names=list(catename2label['catename']), digits=4))

