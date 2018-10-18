import argparse
import pickle
import os
import cv2 as cv
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


IMG_EXTENSIONS = ('.jpg', '.png', '.jpeg')


def open_imgs(dirname, max_items, max_dirs):
    '''
    dirname should be like "dir/innerone/finaldir/",
    always ending with a /
    '''
    imgs = {}
    for innerdir in os.listdir(dirname)[:max_dirs]:
        if os.path.isdir(dirname + innerdir) and '.' not in innerdir:
            for filename in os.listdir(dirname + innerdir)[:max_items]:
                if filename.lower().endswith(IMG_EXTENSIONS):
                    curr = cv.imread(dirname + innerdir + '/' + filename)
                    imgs[dirname + innerdir + '/' + filename] = cv.cvtColor(
                        curr, cv.COLOR_BGR2RGB
                    )
    return imgs


def compute_descs(imgs):
    descs = {}
    surf = cv.xfeatures2d.SURF_create(400)  # value used in docs
    for filename, img in imgs.items():
        _, desc = surf.detectAndCompute(img, None)
        descs[filename] = desc
    return descs


def create_vocab(descs, sz=300):
    scaler = StandardScaler()
    X = []
    for _, value in descs.items():
        for desc in value:
            X.append(desc)
    X_std = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=sz).fit(X_std)
    return kmeans, scaler


def create_hist(desc, vocab):
    kmeans, scaler = vocab
    desc_std = scaler.transform(desc)
    prediction = kmeans.predict(desc_std)
    hist_data = np.zeros(kmeans.n_clusters)
    for pred in prediction:
        hist_data[pred] += 1
    return hist_data


def create_hists(descs, vocab):
    hists = []
    for filename, desc in descs.items():
        hists.append((filename, create_hist(desc, vocab)))
    return hists


def save_data(hists, vocab):
    with open('data/hists.pkl', 'wb') as histsfile:
        pickle.dump(hists, histsfile)

    with open('data/vocab.pkl', 'wb') as vocabfile:
        pickle.dump(vocab, vocabfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treinador da base de dados')
    parser.add_argument('--dir', '-d', dest='dirname', required=True,
                        help='Diretorio contendo diretorios ' +
                        'que contenham imagens (tal como o encontrado no' +
                        'do Caltech101)')
    parser.add_argument('--max-items', '-mi', dest='max_items',
                        default=10, type=int)
    parser.add_argument('--max-dirs', '-md', dest='max_dirs',
                        type=int, default=10,
                        help='Numero maximo de subdiretorios a serem ' +
                        'acessados dentro do definido pelo parametro --dir')
    parser.add_argument('--clusters', '-c', default=300,
                        type=int, dest='n_clusters',
                        help='Numero de clusters usados para a ' +
                        'criacao do vocabulario')
    args = parser.parse_args()

    dirname = args.dirname
    if not dirname.endswith('/'):
        dirname += '/'
    imgs = open_imgs(dirname, args.max_items, args.max_dirs)
    print('Loaded images')
    descs = compute_descs(imgs)
    print('Generated descriptors')
    vocab = create_vocab(descs, args.n_clusters)
    print('Created vocabulary')
    hists = create_hists(descs, vocab)
    print('Created hists')
    save_data(hists, vocab)
    print('Saved data')
    print('Training DONE!')
