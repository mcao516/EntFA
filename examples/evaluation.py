#!/usr/bin/env python
# coding=utf-8
"""
Factuality evaluation using the trained KNN classifier.
"""

import argparse
import os
import json
import torch
import spacy
import pickle
import numpy as np

from os.path import join
from tqdm import tqdm
from fairseq.models.bart import BARTModel

from EntFA.model import ConditionalSequenceGenerator
from EntFA.utils import prepare_cmlm_inputs, prepare_mlm_inputs, get_probability_parallel
from EntFA.utils import read_lines

nlp = spacy.load('en_core_web_sm')


def build_models(args):
    prior_bart = BARTModel.from_pretrained(args.mlm_path,
                                        checkpoint_file='model.pt',
                                        data_name_or_path=args.mlm_path)
    prior_model = ConditionalSequenceGenerator(prior_bart)

    bart = BARTModel.from_pretrained(args.cmlm_model_path,
                                    checkpoint_file='checkpoint_best.pt',
                                    data_name_or_path=args.data_name_or_path)
    model = ConditionalSequenceGenerator(bart)

    return prior_model, model


def extract_features(source, hypothesis, prior_model, model):
    features = []
    empty, error_count = 0, 0

    for index in tqdm(range(len(hypothesis))):
        source_doc, target_doc = source[index], hypothesis[index]
        target_doc = target_doc.replace("“", '"').replace("”", '"').replace("’", "'")
        target_doc = target_doc.replace("%.", "% .")
        target_doc = target_doc.replace("%,", "% ,")
        target_doc = target_doc.replace("%)", "% )")
        
        # extract entities
        ent_parts = nlp(target_doc).to_json()['ents']
        entities = [target_doc[e['start']: e['end']] for e in ent_parts]

        if len(ent_parts) > 0:
            pri_inputs = prepare_mlm_inputs(source, target_doc, ent_parts=ent_parts)
            pos_inputs = prepare_cmlm_inputs(source_doc, target_doc, ent_parts=ent_parts)

            # calculate probability features
            try:
                pri_probs = get_probability_parallel(prior_model, pri_inputs[0], pri_inputs[1], pri_inputs[2], pri_inputs[3], mask_filling=True)
                pos_probs = get_probability_parallel(model, pos_inputs[0], pos_inputs[1], pos_inputs[2], pos_inputs[3])
                
                # overlapping feature
                source_doc = source_doc.lower()
                overlap = []
                for e in entities:
                    if e[:4] == 'the ': e = e[4:]
                    if e.lower() in source_doc:
                        overlap.append(1)
                    else:
                        overlap.append(0)
                
                assert len(pri_probs) == len(pos_probs) == len(pri_inputs[2]) == len(pos_inputs[3])
                features.append((pos_inputs[3], pos_inputs[2], pri_probs, pos_probs, overlap))
            except AssertionError as err:
                print("{}: {}".format(index, err))
                error_count += 1
            
        else:
            empty += 1
            features.append(([], [], [], [], []))
    
    return features


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


def main(args):
    print('- Build prior/posterior models...')
    prior_model, posterior_model = build_models(args)
    print('- Done.')

    print('- Read source documents and summaries...')
    source = read_lines(args.source_path)
    hypothesis = read_lines(args.target_path)
    print('- Done. {} summaries to be evaluated.'.format(len(hypothesis)))

    print('- Extract features...')
    features = extract_features(source, hypothesis, prior_model, posterior_model)
    print('- Done.')

    test_features = []
    for sample in features:
        for pri, pos, ovrp in zip(sample[2], sample[3], sample[4]):
            test_features.append([pri, pos, ovrp])
    
    print('- Start inference...')
    classifier = pickle.load(open(args.knn_model_path, 'rb'))
    Z = infernece(test_features, classifier)
    print('- Done.')

    print('- Total extracted entities: ', Z.shape[0])
    print('- Non-factual entities: {:.2f}%'.format((Z.sum() / Z.shape[0]) * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_path",
        type=str,
        default=None,
        required=True,
        help="The path of the source articles.",
    )
    parser.add_argument(
        "--target_path",
        type=str,
        default=None,
        required=True,
        help="The path of the summaries to be evaluated.",
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
        "--knn_model_path",
        type=str,
        required=True,
    )
    
    args = parser.parse_args()
    main(args)