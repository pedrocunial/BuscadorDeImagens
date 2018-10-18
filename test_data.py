from train_dataset import create_hist

import pickle
import argparse
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_hists(datadir):
    try:
        with open(datadir + 'hists.pkl', 'rb') as histsfile:
            hists = pickle.load(histsfile)
    except IOError as err:
        raise ValueError('Arquivo de treinamento nao encontrado,' +
                         ' voce ja executou o treinamento? (train_dataset.py)')
    return hists


def get_vocab(datadir):
    try:
        with open(datadir + 'vocab.pkl', 'rb') as vocabfile:
            vocab = pickle.load(vocabfile)
    except IOError as err:
        raise ValueError('Arquivo de treinamento nao encontrado,' +
                         ' voce ja executou o treinamento? (train_dataset.py)')
    return vocab


def open_img(imgname):
    img = cv.imread(imgname)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def compute_desc(img):
    surf = cv.xfeatures2d.SURF_create(400)
    return surf.detectAndCompute(img, None)[1]


def compare_hists(hist, hists):
    result = {}
    hist = hist.astype(np.float32)
    for h in hists:
        # h[0] -> filename
        # h[1] -> hist
        result[h[0]] = cv.compareHist(hist, h[1].astype(np.float32),
                                      cv.HISTCMP_CHISQR)
    return sorted([(v, k) for (k, v) in result.items()])


def show_imgs(src, matches, top_n=3):
    fig = plt.figure('Imagem Original')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(src)
    for i in range(top_n):
        fig = plt.figure(f'Match #{i+1}')
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(open_img(matches[i][1]))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Buscador de imagens')
    parser.add_argument('--datadir', '-d', dest='datadir', default='data/',
                        help='Diretorio contendo dados de treinamento')
    parser.add_argument('--image', '--img', '-i', dest='img', required=True,
                        help='Imagem a ser testada (argumento obrigatorio)')

    args = parser.parse_args()

    if not args.datadir.endswith('/'):
        datadir = args.datadir + '/'
    else:
        datadir = args.datadir

    hists = get_hists(datadir)
    vocab = get_vocab(datadir)
    src_img = open_img(args.img)
    src_hist = create_hist(compute_desc(src_img), vocab)
    matches = compare_hists(src_hist, hists)
    show_imgs(src_img, matches)
