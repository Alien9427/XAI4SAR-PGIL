from network import ResNet18_TSX_Comp, ResNet18_TSX, ResNet18_TSX_Comp_CNN, ResNet18_TSX_Comp_CNN2
import numpy as np
import pandas as pd
import torch, argparse, transform_data, warnings, os
from slc_dataset import Comp_Ice_Dataset, Comp_Urban_Dataset
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
    if config['net_type'] == 'CNN':
        model = ResNet18_TSX(config['cate_num'])
    elif config['net_type'] == 'CompNet':
        model = ResNet18_TSX_Comp(config['topic_num'])
    elif config['net_type'] == 'CompCNN':
        # model = ResNet18_TSX_Comp_CNN(config['topic_num'], config['cate_num'])
        model = ResNet18_TSX_Comp_CNN2(config['cate_num'])
    else:
        Exception('please input correct model config')
    model.load_state_dict(torch.load(config['models']['pretrained'], map_location=torch.device('cpu')))
    return model

def get_dataloader(config):
    if 'ICE' in config['datatxt_valid']:
        data_transforms = transforms.Compose([
            transform_data.Normalize_img(mean=0.437279412021903, std=0.03280072512541247), # ICE
            # transform_data.Normalize_img(mean=0.4687830925630793, std=0.045432142815676486),  # Urban
            transform_data.Numpy2Tensor_img(channels=3)
        ])

        dataset_valid = Comp_Ice_Dataset(data_txt=config['datatxt_valid'],
                                         cate_txt=config['cate_txt'],
                                         data_root=config['data_root'],
                                         transform=data_transforms)
    elif 'urban' in config['datatxt_valid']:
        data_transforms = transforms.Compose([
            # transform_data.Normalize_img(mean=0.437279412021903, std=0.03280072512541247),  # ICE
            transform_data.Normalize_img(mean=0.4687830925630793, std=0.045432142815676486),  # Urban
            transform_data.Numpy2Tensor_img(channels=3)
        ])
        dataset_valid = Comp_Urban_Dataset(data_txt=config['datatxt_valid'], cate_txt=config['cate_txt'],
                                         data_root=config['data_root'],
                                         transform=data_transforms)
    elif 'Paris' in config['datatxt_valid']:
        data_transforms = transforms.Compose([
            transform_data.Normalize_img(mean=2.7738790337190156, std=0.756960995415545),
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
    if 'ICE' in config['datatxt_valid']:
        data_transforms = transforms.Compose([
            transform_data.Normalize_img(mean=0.437279412021903, std=0.03280072512541247),  # ICE
            # transform_data.Normalize_img(mean=0.4687830925630793, std=0.045432142815676486),  # Urban
            transform_data.Numpy2Tensor_img(channels=3)
        ])

        dataset_train = Comp_Ice_Dataset(data_txt=train_txt, cate_txt=config['cate_txt'],
                                         data_root=config['data_root'],
                                         # data_root = '../data/Annotate_0417/',
                                         transform=data_transforms)

        dataset_valid = Comp_Ice_Dataset(data_txt=config['datatxt_valid'], cate_txt=config['cate_txt'],
                                         data_root=config['data_root'],
                                         transform=data_transforms)
    elif 'urban' in config['datatxt_valid']:
        data_transforms = transforms.Compose([
            # transform_data.Normalize_img(mean=0.437279412021903, std=0.03280072512541247),  # ICE
            transform_data.Normalize_img(mean=0.4687830925630793, std=0.045432142815676486),  # Urban
            transform_data.Numpy2Tensor_img(channels=3)
        ])
        dataset_train = Comp_Urban_Dataset(data_txt=train_txt, cate_txt=config['cate_txt'],
                                         data_root=config['data_root'],
                                         transform=data_transforms)

        dataset_valid = Comp_Urban_Dataset(data_txt=config['datatxt_valid'], cate_txt=config['cate_txt'],
                                         data_root=config['data_root'],
                                         transform=data_transforms)
    elif 'Paris' in config['datatxt_valid']:
        data_transforms = transforms.Compose([
            transform_data.Normalize_img(mean=2.7738790337190156, std=0.756960995415545),
            transform_data.Numpy2Tensor_img(channels=3)
        ])
        dataset_train = Comp_Ice_Dataset(data_txt=train_txt, cate_txt=config['cate_txt'],
                                         data_root=config['data_root'],
                                         # data_root = '../data/Annotate_0417/',
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

def CosSimilarity_Test(config, train_txt):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_pretrained_model(config)
    model.to(device)
    model.eval()

    dataloader = get_dataloader_SVM(config, train_txt)

    embs_arr = np.zeros([0, 256])
    label_arr = np.zeros([0, ])

    for sample in dataloader['train']:
        data = sample['img']
        embs = model(data.to(device))
        label_arr = np.concatenate((label_arr, sample['label'].numpy()), axis=0) # shape: N,
        embs_arr = np.concatenate((embs_arr, embs.cpu().data.numpy()), axis=0) # shape: N * 50

    embs_train = {}
    train_labels = np.unique(label_arr)
    for label in train_labels:
        embs_train[label] = np.mean(embs_arr[label_arr == label,], axis=0)

    D = torch.nn.CosineSimilarity(dim=0)

    pred_labels = []
    true_labels = []

    all_num = 0
    acc_num = 0
    acc_valid = 0


    for sample in dataloader['valid']:

        data = sample['img']
        labels = sample['label'].to(device)
        D_result = torch.zeros([len(labels), len(embs_train.keys())])
        embs_valid = model(data.to(device))
        true_labels = true_labels + list(np.array(labels.data.cpu()))

        for label in embs_train.keys():
            D_result[:, int(label)] = torch.Tensor([D(torch.Tensor(embs_train[label]), embs_valid.cpu().data[j,]) for j in range(len(labels))])

        pred_labels = pred_labels + list(torch.argsort(D_result, 1, True)[:,0])

    return np.array(pred_labels), np.array(true_labels)

def CNN_test(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_pretrained_model(config)

    dataloader = get_dataloader(config)

    model.to(device)
    model.eval()
    pred_labels = []
    true_labels = []
    test_paths = []

    with torch.no_grad():
        for sample in dataloader['valid']:
            data = sample['img']
            labels = sample['label'].to(device)
            output = model(data.to(device))
            test_paths += sample['scat_path']

            _, pred = torch.Tensor.max(output, 1)

            pred_labels = pred_labels + list(np.array(torch.squeeze(pred).data.cpu()))
            true_labels = true_labels + list(np.array(labels.data.cpu()))

            del labels, output, pred

    # label2name = pd.read_csv(config['cate_txt'])
    # pred_catename = [label2name.loc[pred_labels[i]]['catename'] for i in range(len(pred_labels))]
    # df = pd.DataFrame(columns=['path', 'catename'])
    # df['path'] = pd.Series(test_paths)
    # df['catename'] = pd.Series(pred_catename)
    # df.to_csv('/home/hzl/STAT/pytorch_code/ICE/data/GF3_Paris_cnn_retrain.txt', index=False)
    return np.array(pred_labels), np.array(true_labels)

def Comp_CNN_test(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_pretrained_model(config)
    # for comp_cnn_train_3 model (comp_cnn2), compNet needed
    model_CompNet = ResNet18_TSX_Comp(config['topic_num'])
    model_CompNet.load_state_dict(torch.load('../model/ICE_comp_n1_soft_topic175_retrain.pth'))
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

def Comp_CNN_test2(config, train_txt):
    """
    计算所有训练样本的平均attr
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_pretrained_model(config)
    lda_model = LdaModel.load('../result/ICE_lda_175.pkl')
    kmeans_model = joblib.load('../result/ICE_kmeans.pkl')

    dataloader = get_dataloader_SVM(config, train_txt)

    model.to(device)
    model.eval()
    pred_labels = []
    true_labels = []

    # attr_arry = torch.zeros([0, config['topic_num']])
    attr_arry = torch.zeros([0, 256])
    label_arry = torch.zeros([0, ])

    # ICE topic175 layer3-feat ss:
    lambda_ice = torch.Tensor(
        [0.21489193057677594, 0.2161175891189735, 0.146607673825186, -0.07945569604899573, 0.12972769303356757,
         0.1891051712646619, 0.06335676688612851])


    lambda_ice = (lambda_ice - lambda_ice.min()) / (lambda_ice.max() - lambda_ice.min())


    with torch.no_grad():
        for sample in dataloader['train']:
            data = sample['img']
            scat_paths = sample['scat_path']
            # labels = sample['label'].to(device)
            embs, output = model(data.to(device))
            # attr = get_BoT(scat_paths, kmeans_model, lda_model, config['topic_num'])  # shape: batchsize * topic_num
            # attr_arry = torch.cat((attr_arry, torch.Tensor(attr)), axis=0)  # shape: N * 50 # 用BoT的度量计算sim
            attr_arry = torch.cat((attr_arry, torch.Tensor(embs.cpu())), axis=0) # 用layer3的度量计算sim
            label_arry = torch.cat((label_arry, sample['label']), axis=0)  # shape: N,


        # step 1: 计算训练集中每类样本的平均topic分布
        label_set = torch.unique(label_arry, sorted=True)
        avg_attr = [torch.mean(attr_arry[label_arry == i,], axis=0) for i in label_set]

        for sample in dataloader['valid']:
            data = sample['img']
            labels = sample['label'].to(device)
            scat_paths = sample['scat_path']
            # attr = get_BoT(scat_paths, kmeans_model, lda_model, config['topic_num'])  # shape: batchsize * topic_num

            embs, output = model(data.to(device))

            sim = torch.zeros([len(labels), len(label_set)])
            for i in range(len(labels)):
                # sim[i,] = torch.Tensor(
                #     [func.cosine_similarity(avg_attr[j], torch.Tensor(attr[i,]), dim=0) for j in range(len(label_set))]) # BoT
                sim[i,] = torch.Tensor(
                    [func.cosine_similarity(avg_attr[j], embs.cpu()[i,], dim=0) for j in range(config['cate_num'])]) # layer3-feat

                # sim[i, label[i]] = 0

            output = output + lambda_ice.cuda() * sim.cuda()
            _, pred = torch.Tensor.max(output, 1)

            pred_labels = pred_labels + list(np.array(torch.squeeze(pred).data.cpu()))
            true_labels = true_labels + list(np.array(labels.data.cpu()))

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

def SVM_topics(config, train_txt):
    dataloader = get_dataloader_SVM(config, train_txt)

    lda_model = LdaModel.load('../result/ICE_lda_175.pkl')
    kmeans_model = joblib.load('../result/ICE_kmeans.pkl')

    topics_arr = np.zeros([0, config['topic_num']])
    label_arr = np.zeros([0, ])
    for sample in dataloader['train']:
        scat_paths = sample['scat_path']
        attr = get_BoT(scat_paths, kmeans_model, lda_model, config['topic_num'])  # shape: batchsize * topic_num
        label_arr = np.concatenate((label_arr, sample['label'].numpy()), axis=0)  # shape: N,
        topics_arr = np.concatenate((topics_arr, attr), axis=0)  # shape: N * 50

    embs_min = np.min(topics_arr)
    embs_max = np.max(topics_arr)

    embs_arr = (topics_arr - embs_min) / (embs_max - embs_min)
    classifier = SVC(kernel='rbf', class_weight='balanced', probability=True)
    classifier.fit(embs_arr, label_arr)

    pred_labels = []
    true_labels = []

    for sample in dataloader['valid']:
        scat_paths = sample['scat_path']
        attr = get_BoT(scat_paths, kmeans_model, lda_model, config['topic_num'])  # shape: batchsize * topic_num
        labels = sample['label']

        true_labels = true_labels + list(np.array(labels.data))
        attr = (attr - embs_min) / (embs_max - embs_min)

        pred_labels = pred_labels + list(np.argsort(classifier.decision_function(attr))[:, -1])
        del labels

    return np.array(pred_labels), np.array(true_labels)

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

def save4feature(config, save_name):
    os.environ["CUDA_VISIBLE_DEVICES"] = config['device']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name())


    model = load_pretrained_model(config)

    # for comp_cnn_train_3 model (comp_cnn2), compNet needed
    # model_CompNet = ResNet18_TSX_Comp(config['topic_num'])
    # model_CompNet.load_state_dict(torch.load('../model/ICE_comp_n1_soft_topic175_retrain.pth'))
    # model_CompNet.to(device)
    # model_CompNet.eval()

    dataloader = get_dataloader(config)
    lda_model = LdaModel.load('../result/GF3_Paris_lda_200.pkl')
    kmeans_model = joblib.load('../result/GF3_Paris_kmeans.pkl')

    model.to(device)
    model.eval()
    outfeat = []
    true_labels = []
    data_path = []

    with torch.no_grad():
        for sample in dataloader['valid']:
            data = sample['img']
            labels = sample['label'].to(device)
            path = sample['scat_path']

            # comp_feat = model_CompNet(data.to(device))
            # feat = model(data.to(device), comp_feat) # comp_cnn_2
            feat = model(data.to(device))
            # feat = get_BoT(path, kmeans_model, lda_model, config['topic_num'])
            # print(feat.shape)
            true_labels = true_labels + list(np.array(labels.data.cpu()))
            outfeat = outfeat + list(np.array(torch.squeeze(feat).data.cpu()))
            # outfeat = outfeat + list(np.squeeze(feat))
            data_path = data_path + path

            del labels, feat

    outfeat = np.array(outfeat)
    true_labels = np.array(true_labels)

    df_return = pd.DataFrame(data=outfeat, index=true_labels)
    df_return['label'] = pd.Series(true_labels, index=true_labels)
    df_return['path'] = pd.Series(data_path, index=true_labels)

    df_return.to_csv(save_name, index=False)

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

def feature_vis(config, feat_path):
    data = pd.read_csv(feat_path)
    feature, label, path = get_score_label(data)

    label2name = pd.read_csv(config['cate_txt'])

    X_tsne = TSNE(learning_rate=50, n_components=2).fit_transform(feature)
    x_min, x_max = np.min(X_tsne, axis=0), np.max(X_tsne, axis=0)
    X = (X_tsne - x_min) / (x_max - x_min)

    colors = ['#FFC312', '#C4E538', '#12CBC4', '#FDA7DF', '#ED4C67', '#006266', '#1B1464', '#5758BB',
              '#F97F51', '#1B9CFC']
    hover = HoverTool(tooltips=[
        ('path', '@path')
    ])

    p = figure(plot_width=800, plot_height=600, tools=[hover, 'save', 'pan', 'wheel_zoom'], toolbar_location="right")

    for i in set(label):
        source = ColumnDataSource(data=dict(x=[k for (j, k) in enumerate(X[:, 0]) if label[j] == i],
                                            y=[k for (j, k) in enumerate(X[:, 1]) if label[j] == i],
                                            path=[k for (j, k) in enumerate(path) if label[j] == i]))

        p.circle('x', 'y', source=source, size=10,
                 color='black',
                 line_width=0.7,
                 fill_color=colors[i],
                 legend_label=label2name.loc[i]['catename']
                 )

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
    p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    p.outline_line_width = 1
    p.outline_line_color = "black"
    p.x_range = Range1d(0, 1.5)
    p.legend.label_text_color = 'black'
    p.legend.label_text_font_size = '12pt'
    show(p)

def feature_vis_kmeans(feature, label, pred, path, config):
    X_tsne = TSNE(learning_rate=50, n_components=2).fit_transform(feature)
    x_min, x_max = np.min(X_tsne, axis=0), np.max(X_tsne, axis=0)
    X = (X_tsne - x_min) / (x_max - x_min)
    label2name = pd.read_csv(config['cate_txt'])

    colors = ['#FFC312', '#C4E538', '#12CBC4', '#FDA7DF', '#ED4C67', '#006266', '#1B1464', '#5758BB']
    colors_kmeans = ['#F97F51', '#1B9CFC', '#F8EFBA', '#58B19F', '#2C3A47', '#B33771', '#3B3B98', '#D6A2E8', '#a4c639', '#ffbf00', '#9966cc', '#008000']

    hover = HoverTool(tooltips=[
        ('path', '@path')
    ])

    # 1. draw labeled data
    p = figure(plot_width=800, plot_height=600, tools=[hover, 'save', 'pan', 'wheel_zoom'], toolbar_location="right")

    for i in set(label):
        source = ColumnDataSource(data=dict(x=[k for (j, k) in enumerate(X[:, 0]) if label[j] == i],
                                            y=[k for (j, k) in enumerate(X[:, 1]) if label[j] == i],
                                            path=[k for (j, k) in enumerate(path) if label[j] == i]))

        p.circle('x', 'y', source=source, size=10,
                 color='black',
                 line_width=0.7,
                 fill_color=colors[i],
                 legend_label=label2name.loc[i]['catename']
                 )
    # 1. draw kmeans predict data
    p_kmeans = figure(plot_width=800, plot_height=600, tools=[hover, 'save', 'pan', 'wheel_zoom'], toolbar_location="right")

    for i in set(pred):
        source_kmeans = ColumnDataSource(data=dict(x=[k for (j, k) in enumerate(X[:, 0]) if pred[j] == i],
                                                   y=[k for (j, k) in enumerate(X[:, 1]) if pred[j] == i],
                                                   path=[k for (j, k) in enumerate(path) if pred[j] == i]))
        p_kmeans.circle('x', 'y', source=source_kmeans, size=10,
                 color='black',
                 line_width=0.7,
                 fill_color=colors_kmeans[i],
                 legend_label=str(i)
                 )

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
    p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    p.outline_line_width = 1
    p.outline_line_color = "black"
    p.x_range = Range1d(0, 1.5)
    p.legend.label_text_color = 'black'
    p.legend.label_text_font_size = '12pt'

    p_kmeans.xgrid.grid_line_color = None
    p_kmeans.ygrid.grid_line_color = None
    p_kmeans.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p_kmeans.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p_kmeans.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p_kmeans.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    p_kmeans.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
    p_kmeans.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
    p_kmeans.outline_line_width = 1
    p_kmeans.outline_line_color = "black"
    p_kmeans.x_range = Range1d(0, 1.5)
    p_kmeans.legend.label_text_color = 'black'
    p_kmeans.legend.label_text_font_size = '12pt'
    subplot = row([p, p_kmeans])
    show(subplot)

def label_vis_kmeans(pred, label, config):
    label2name = pd.read_csv(config['cate_txt'])
    subplots = []
    colors = Pastel1[len(label2name)]
    for i in set(pred):
        i_idx = np.where(pred == i)[0]
        i_label = label[i_idx]
        pie_data = {}
        for j in set(i_label):
            pie_data[label2name.loc[j]['catename']] = np.sum(i_label == j)

        pie_df = pd.Series(pie_data).reset_index(name='value').rename(columns={'index': 'class'})
        pie_df['angle'] = pie_df['value'] / pie_df['value'].sum() * 2 * np.pi
        # pie_df['color'] = Pastel1[len(pie_data)]
        pie_df['color'] = [colors[int(label2name.loc[label2name['catename'] == pie_df.loc[j]['class']]['label'])] for j
                           in range(len(pie_df))]

        p = figure(plot_height=200, title="Kmeans Class " + str(i), toolbar_location=None,
                   tools="hover", tooltips="@class: @value", x_range=(-0.5, 1.0))

        p.wedge(x=0, y=1, radius=0.3,
                start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                line_color="black", fill_color='color', legend_field='class', source=pie_df)

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None

        subplots.append(p)

    grid = gridplot([[subplots[0], subplots[1], subplots[2], subplots[3]],
                     [subplots[4], subplots[5], subplots[6], subplots[7]],
                     [subplots[8], subplots[9], subplots[10], subplots[11]],
                     [subplots[12], subplots[13], None, None]], plot_width=380, plot_height=250)
    show(grid)


def save_feature_metrics(feat_path, datatxt_train, save_file):
    data = pd.read_csv(feat_path)
    feature, label, path = get_score_label(data)
    ss_all = metrics.silhouette_samples(feature, label, metric = 'euclidean')
    train_data = pd.read_csv(datatxt_train)
    train_data['ss'] = ''

    # ss_all 归一化
    ss_all[ss_all > 0] = ss_all[ss_all > 0] / ss_all.max()
    ss_all[ss_all < 0] = ss_all[ss_all < 0] / -ss_all.min()

    for i, each_path in enumerate(path):
        path_name = each_path.split('_0417/')[1].split('_scat')[0]
        train_data.loc[train_data['path'] == path_name, 'ss'] = ss_all[i]

    train_data.to_csv(save_file)

def feature_metrics(config, feat_path):
    data = pd.read_csv(feat_path)
    feature, label, path = get_score_label(data)

    label2name = pd.read_csv(config['cate_txt'])

    ss = metrics.silhouette_score(feature, label, metric='euclidean')
    ch = metrics.calinski_harabasz_score(feature, label)
    ss_all = metrics.silhouette_samples(feature, label, metric = 'euclidean')

    intra_dist = 0
    inter_dist = 0
    all_center = np.sum(feature, axis=0) / feature.shape[0]
    for each_label in set(list(label)):
        feat_sel = feature[np.where(label == each_label)[0],]
        feat_center = np.sum(feat_sel, axis=0)/feat_sel.shape[0]
        intra_dist += np.sum((feat_sel - feat_center) * (feat_sel - feat_center), axis=0) / feature.shape[0]
        inter_dist += ((feat_center - all_center) * (feat_center - all_center)) * feat_sel.shape[0] / feature.shape[0]

    ss_return = []
    print('calinski-harabasz metric: ', ch)
    print('silhouette coefficient (all): ', ss)
    for i in range(config['cate_num']):
        print('silhouette coefficient (class %s)： %f'%(label2name.loc[i]['catename'], np.sum(ss_all[np.where(label == i)]) / np.where(label == i)[0].shape[0]))
        ss_return.append(np.sum(ss_all[np.where(label == i)]) / np.where(label == i)[0].shape[0])
        print('intra class dist (class %s): %f'%(label2name.loc[i]['catename'], intra_dist[i]))
        print('inter class dist (class %s): %f' % (label2name.loc[i]['catename'], inter_dist[i]))
    print('intra class dist: ', np.sum(intra_dist))
    print('inter class dist: ', np.sum(inter_dist))
    print('ratio of inter and intra: ', np.sum(inter_dist) / np.sum(intra_dist))

    # return ss_return


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if cmap is None:
        cmap = plt.get_cmap('Oranges')

    norm = mpl.colors.Normalize(vmin=0, vmax=cm_norm.max())

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    plt.imshow(cm_norm, interpolation='nearest', cmap=cmap, norm=norm)
    plt.title(title)

    plt.colorbar()

    # plt.ylim(-1, len(target_names))

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)


    thresh = cm_norm.max() / 1.5

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            if cm_norm[i, j] >= 0.0001:
                plt.text(j, i, "{:0.4f}".format(cm_norm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm_norm[i, j] > thresh else "black")
            else:
                plt.text(j, i, 0,
                         horizontalalignment="center",
                         color="white" if cm_norm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(int(cm[i, j])),
                     horizontalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
        # if cm[i,j] == 0:
        #     plt.text(j, i, 0,
        #              horizontalalignment="center",
        #              color="white" if cm[i, j] > thresh else "black")
        # else:
        #     plt.text(j, i, "{:0.4f}".format(cm[i, j]),
        #          horizontalalignment="center",
        #          color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


    plt.show()

def vis_bars(config):
    BoT_data = pd.read_csv('../result/ICE_BoT_topic175_retrain_traindata.txt')
    BoT_feature, BoT_label, BoT_path = get_score_label(BoT_data)

    # data = pd.read_csv('../result/ICE_compfc_topic175_retrain_traindata.txt')
    # feature, label, path = get_score_label(data)
    #
    # data_hard = pd.read_csv('../result/ICE_compfc_topic175_retrain_traindata_hard.txt')
    # feature_hard, label_hard, path_hard = get_score_label(data_hard)
    #
    # kmeans = joblib.load('../result/ICE_BoTtrain_topic175_kmeans_12.pkl')  # BoT的k-means模型

    # pred = []
    # for each_path in path:
    #     pred.append(kmeans.predict(BoT_feature[BoT_path.index(each_path)].reshape(1, -1))[0])
    # pred = np.array(pred)
    #
    # pred_hard = []
    # for each_path in path_hard:
    #     pred_hard.append(kmeans.predict(BoT_feature[BoT_path.index(each_path)].reshape(1, -1))[0])
    # pred_hard = np.array(pred_hard)

    catename2label = pd.read_csv(config['cate_txt'])

    for i in set(BoT_label):
        BoT_center = np.mean(BoT_feature[BoT_label == i], axis=0)
        x_axis = np.arange(len(BoT_center))
        # plt.figure(i)
        # plt.bar(x_axis, BoT_center, label=catename2label.loc[catename2label['label'] == i]['catename'][i])
        #
        # plt.ylim((0, 1))
        # plt.legend()

        print(catename2label.loc[catename2label['label'] == i]['catename'][i])
        print(np.argsort(BoT_center)[-5:])
        print(np.sort(BoT_center)[-5:])
        # print(np.where(BoT_center >= 0.001)[0])
        # print(BoT_center[BoT_center > 0.001])

    # BoT_center_0 = kmeans.cluster_centers_[cluster_label]
    #
    # feat_0 = feature[pred == cluster_label]
    # feat_0_softmax = func.softmax(torch.Tensor(feat_0), 1)
    # feat_0_cluster = torch.mean(feat_0_softmax, axis=0)
    #
    # feat_hard_0 = feature_hard[pred_hard == cluster_label]
    # feat_hard_0_softmax = func.softmax(torch.Tensor(feat_hard_0), 1)
    # feat_hard_0_cluster = torch.mean(feat_hard_0_softmax, axis=0)

    # idx = (BoT_center_0 > 0.00001)
    # x_axis=np.arange(np.sum(idx))
    # labels = list(np.where(BoT_center_0 > 0.00001)[0])
    #
    # plt.figure(1)
    # # plt.bar(x_axis, BoT_center_0[idx], width=0.3, label='BoT')
    # plt.bar(x_axis + 0.3, feat_hard_0_cluster[idx], width=0.3, label='hard', tick_label=labels)
    # # plt.bar(x_axis + 0.6, feat_0_cluster[idx], width=0.3, label='soft')
    # plt.ylim((0,1))

    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ICE_test')
    parser.add_argument('--pretrained', default='../model/ICE_cnn_ds1_layer4_pretrain_8178.pth') #
    parser.add_argument('--net_type', default='CNN') # CNN, CompNet, CompCNN
    parser.add_argument('--data_root', default='../data/Annotate_0417/') # GF3_Paris_annotate
    parser.add_argument('--datatxt_valid', default='../data/1_ICE_dataset_c7_train45.txt') # ICE_dataset_test_n50_c7
    parser.add_argument('--cate_txt', default='../data/ICE_catename2label_7.txt')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', default='1')
    parser.add_argument('--cate_num', type=int, default=7)
    parser.add_argument('--topic_num', type=int, default=200)

    args = parser.parse_args()
    config = parameter_setting(args)
    # model = load_pretrained_model(config)
    print(config['models']['pretrained'])

    # print(topic_sparsity(config))

    """ Cos-sim Test
    """
    # pred_labels, true_labels = CosSimilarity_Test(config, '../data/1_GF3_Paris_dataset_train45.txt')
    # matrix = matrix_analysis(pred_labels, true_labels, config['cate_num'])
    # catename2label = pd.read_csv(config['cate_txt'])
    # # print(catename2label)
    # print(matrix)
    # print(metrics.classification_report(true_labels, pred_labels, target_names=list(catename2label['catename'])))

    """ CNN Test
    """
    # pred_labels, true_labels = CNN_test(config)
    # # pred_labels, true_labels = Comp_CNN_test2(config, '../data/1_ICE_dataset_c7_train45.txt')
    # # pred_labels, true_labels = Comp_CNN_test(config)
    # matrix = matrix_analysis(pred_labels, true_labels, config['cate_num'])
    # catename2label = pd.read_csv(config['cate_txt'])
    # # print(catename2label)
    # print(matrix)
    # print(metrics.classification_report(true_labels, pred_labels, target_names=list(catename2label['catename']), digits=4))


    """ CompNet Feature Vis
    """
    save4feature(config, '../result/ICE_cnn_ds1_layer4_pretrain_8178_traindata.txt')
    # feature_vis(config, '../result/GF3_Paris_n20_traindata_topic.txt')

    """ SVM Test
    """
    # pred_labels, true_labels = SVM_test(config, '../data/GF3_Paris_dataset_gai_train_num_20.txt')
    # # pred_labels, true_labels = SVM_topics(config, '../data/1_ICE_dataset_c7_train45.txt')
    # matrix = matrix_analysis(pred_labels, true_labels, config['cate_num'])
    # catename2label = pd.read_csv(config['cate_txt'])
    # # print(catename2label)
    # print(matrix)
    # print(metrics.classification_report(true_labels, pred_labels, target_names=list(catename2label['catename']), digits=4))

    """ Feature metrics
    """
    # feature_metrics(config, feat_path='../result/ICE_layer3_topic100_1.0_retrain.txt')
    # print(ss_return)
    # save_feature_metrics(feat_path='../result/ICE_layer3_topic175_retrain_train.txt',
    #                      datatxt_train='../data/1_ICE_dataset_c7_train45.txt',
    #                      save_file='../data/1_ICE_dataset_c7_train45_ss.txt')
    # ICE topic175 BoT ss: [0.17890033519155477, 0.1712491537810747, 0.017456074980502386, -0.07461694682695726, -0.0849654910118409, -0.011747633908876799, -0.10897659486179585]
    # ICE topic175 layer3-feat ss: [0.21489193057677594, 0.2161175891189735, 0.146607673825186, -0.07945569604899573, 0.12972769303356757, 0.1891051712646619, 0.06335676688612851]

    """ Kmeans with silhouette metrics
    """
    data = pd.read_csv('../result/ICE_cnn_ds1_layer4_pretrain_8178_traindata.txt')
    feature, label, path = get_score_label(data)
    #
    BoT_data = pd.read_csv('../result/ICE_BoT_topic175_retrain_traindata.txt')
    BoT_feature, _, BoT_path = get_score_label(BoT_data)
    #
    # label2name = pd.read_csv(config['cate_txt'])

    # for k in range(8,15):
    #     kmeans = KMeans(n_clusters=k)
    #     kmeans_model = kmeans.fit(feature)
    #     joblib.dump(kmeans_model, '../result/ICE_BoTtrain_topic175_kmeans_' + str(k) + '.pkl')
    #     pred = kmeans.predict(feature)
    #     ss = metrics.silhouette_score(feature, pred, metric='euclidean')
    #     print('k=' + str(k) + ' and SC=' + str(ss))
    #     feature_vis_kmeans(feature, label, pred, path, config)

    kmeans = joblib.load('../result/ICE_BoTtrain_topic175_kmeans_12.pkl') # BoT的k-means模型
    pred = []
    for each_path in path:
        pred.append(kmeans.predict(BoT_feature[BoT_path.index(each_path)].reshape(1,-1))[0])
    pred = np.array(pred)
    # pred = kmeans.predict(feature)
    ss = metrics.silhouette_score(feature, pred, metric='euclidean')
    print('SC=' + str(ss))
    feature_vis_kmeans(feature, label, pred, path, config)

    # label_vis_kmeans(pred, label, config)

    # vis_bars(config)

    """ confusion matrix visualization
    """
    # plot_confusion_matrix(matrix,
    #                       catename2label['catename'],
    #                       title='PA feature + SVM',
    #                       cmap=None,
    #                       normalize=False)