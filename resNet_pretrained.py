import argparse
import os
import time

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize



#models.vgg.model_urls["vgg16"] = "http://webia.lip6.fr/~robert/cours/rdfia/vgg16-397923af.pth"
#os.environ["TORCH_MODEL_ZOO"] = "/tmp/torch"
PRINT_INTERVAL = 50
CUDA = False


mu= [0.485,0.456,0.406]
sigma=[0.299,0.224,0.225]

def get_dataset(batch_size, path):

    # Cette fonction permet de recopier 3 fois une image qui
    # ne serait que sur 1 channel (donc image niveau de gris)
    # pour la "transformer" en image RGB. Utilisez la avec
    # transform.Lambda
    def duplicateChannel(img):
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
        return img

    train_dataset = datasets.ImageFolder(path+'/train',
        transform=transforms.Compose([ # TODO Pré-traitement à faire
            transforms.Resize((224,224)),
            transforms.Lambda(duplicateChannel),
            transforms.ToTensor(), 
            transforms.Normalize(mu, sigma)

        ]))
    val_dataset = datasets.ImageFolder(path+'/test',
        transform=transforms.Compose([ # TODO Pré-traitement à faire
            transforms.Resize((224,224)),
            transforms.Lambda(duplicateChannel),
            transforms.ToTensor(), 
            transforms.Normalize(mu, sigma)
        ]))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=batch_size, shuffle=False, pin_memory=CUDA, num_workers=2)

    return train_loader, val_loader


def extract_features(data, model):
    # TODO init features matrices
    X = []
    y = []
    for i, (input, target) in enumerate(data):
        if i % PRINT_INTERVAL == 0:
            print('Batch {0:03d}/{1:03d}'.format(i, len(data)))
        if CUDA:
            input = input.cuda()
            d = model.forward(input)
            d = d.detach().cpu()
        elif (CUDA==False):
            d = model.forward(input)
            d = d.detach()
     
        d= normalize(d)  
        for e in range(len(d)):     
        	X.append(d[e])        
        	y.append(target[e])

    return np.array(X), np.array(y)


def main(params):
    print('Instanciation de squeezeNet')
    squeezenet = models.squeezenet1_0(pretrained=True)    
    
    print('Instanciation de ResNetrelu7')
    
    #todo ajouter classe
    class SqueezeNetrelu7(nn.Module):
        def __init__(self):
            super(SqueezeNetrelu7, self).__init__()
            # recopier toute la partie convolutionnelle
            self.features = nn.Sequential(*list(squeezenet.features.children()))
            # garder une partie du classifieur, -2 pour s'arrêter à relu7
            self.classifier = nn.Sequential(*list(squeezenet.classifier.children()))
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x
    
    model = SqueezeNetrelu7()
    
    model.eval()
    if CUDA: # si on fait du GPU, passage en CUDA
        model = model.cuda()

    # On récupère les données
    print('Récupération des données')
    train, test = get_dataset(params.batch_size, params.path)

    # Extraction des features
    print('Feature extraction')
    X_train, y_train = extract_features(train, model)
    X_test, y_test = extract_features(test, model)


    # TODO Apprentissage et évaluation des SVM à faire
    print('Apprentissage des SVM')
    svm = LinearSVC(C=0.001)
    svm.fit(X_train, y_train)
    accuracy_a = svm.score(X_train, y_train)
    accuracy_t = svm.score(X_test, y_test)
    print("accuracy apprentissage = ", accuracy_a)
    print("accuracy test = ", accuracy_t)

if __name__ == '__main__':

    # Paramètres en ligne de commande
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='15SceneData', type=str, metavar='DIR', help='path to dataset')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 8)')
    parser.add_argument('--cuda', dest='cuda', action='store_true', help='activate GPU acceleration')

    args = parser.parse_args()
    if args.cuda:
        CUDA = True
        cudnn.benchmark = True

    with torch.no_grad():
        main(args)
    

    input("done")
