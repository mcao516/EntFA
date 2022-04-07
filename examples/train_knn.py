#!/usr/bin/env python
# coding=utf-8
"""
Train a KNN classifier for factuality evaluation.
"""

import argparse
import os
import json
import torch
import pickle
import numpy as np

from tqdm import tqdm
from sklearn import neighbors
from sklearn.metrics import classification_report, f1_score, accuracy_score

from fairseq.models.bart import BARTModel

from EntFA.model import ConditionalSequenceGenerator
from EntFA.utils import prepare_cmlm_inputs, prepare_mlm_inputs, get_probability_parallel


def build_classifier(train_features, train_labels, n=30):
    classifier = neighbors.KNeighborsClassifier(n_neighbors=30, algorithm='auto')
    
    x_mat = np.array(train_features)
    stds = [np.std(x_mat[:, 0]), np.std(x_mat[:, 1]), np.std(x_mat[:, 2])]
    x_mat = np.vstack([x_mat[:, 0]/stds[0],  x_mat[:, 1]/stds[1], x_mat[:, 2]/stds[2]]).transpose()
    y_vec = np.array(train_labels)
    classifier.fit(x_mat, y_vec)
    
    return classifier


def infernece(test_features, classifier):
    """
    Args:
        test_features (List[List]): [[prior, posterior, overlap_feature], ...]
        classifier: KNN classifier
    """
    x_mat = np.array(test_features)
    stds = [np.std(x_mat[:, 0]), np.std(x_mat[:, 1]), np.std(x_mat[:, 2])]
    x_mat = np.vstack([x_mat[:, 0]/stds[0],  x_mat[:, 1]/stds[1], x_mat[:, 2]/stds[2]]).transpose()
    Z = classifier.predict(x_mat)
    return Z


def get_features(data_set, prior_model, model):
    label_mapping = {
        'Non-hallucinated': 0,
        'Factual Hallucination': 0,
        'Non-factual Hallucination': 1
    }

    features, labels = [], []
    for t in tqdm(data_set):
        source, prediction, entities = t['source'], t['prediction'], t['entities']

        inputs = prepare_mlm_inputs(source, prediction, ent_parts=entities)
        priors = get_probability_parallel(prior_model, inputs[0], inputs[1], inputs[2], inputs[3], mask_filling=True)

        inputs = prepare_cmlm_inputs(source, prediction, ent_parts=entities)
        posteriors = get_probability_parallel(model, inputs[0], inputs[1], inputs[2], inputs[3])

        overlaps = [1. if e['ent'].lower() in source.lower() else 0. for e in entities]
        assert len(priors) == len(posteriors) == len(overlaps)

        for i, e in enumerate(entities):
            if label_mapping.get(e['label'], -1) != -1:
                features.append((priors[i], posteriors[i], overlaps[i]))
                labels.append(label_mapping[e['label']])

    return features, labels


def main(args):
    # 1. load training & test dataset
    train_set = json.load(open(args.train_path, 'r'))

    # 2. load weights
    bart = BARTModel.from_pretrained(args.cmlm_model_path,
                                     checkpoint_file='checkpoint_best.pt',
                                     data_name_or_path=args.data_name_or_path)
    prior_bart = BARTModel.from_pretrained(args.mlm_path,
                                           checkpoint_file='model.pt',
                                           data_name_or_path=args.mlm_path)

    # 3. build model
    model = ConditionalSequenceGenerator(bart)
    prior_model = ConditionalSequenceGenerator(prior_bart)

    # 4. training
    train_features, train_labels = get_features(train_set, prior_model, model)
    classifier = build_classifier(train_features, train_labels, n=30)

    # 5. evaluation
    if args.test_path:
        test_set = json.load(open(args.test_path, 'r'))

        test_features, test_labels = get_features(test_set, prior_model, model)
        Z = infernece(test_features, classifier)

        print('accuracy: {:.4}\n\n'.format(accuracy_score(test_labels, Z)))
        print(classification_report(test_labels, Z, target_names=['Factual', 'Non-Factual'], digits=4))

    # 6. save
    save_path = os.path.join(args.output_dir, 'knn_classifier.pkl')
    pickle.dump(classifier, open(save_path, 'wb'))
    print('- model is saved at: ', save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        required=True,
        help="The path of the training set.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="The path of the test set.",
    )
    parser.add_argument(
        "--cmlm_model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mlm_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='.',
    )
    
    args = parser.parse_args()
    main(args)