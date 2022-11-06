import os
import json
import torch


def read_lines(file_path):
    files = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            files.append(line.strip())
    return files


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def get_probability(position, tokens, probs, entity):
    """Calculate the probability of a span.

    Args:
        position: (start, end)
        tokens: ['The', ' Archbishop', ' of', ...]
        probs: [0.50, 0.49, 0.88, ...]
        entity: Rodgers
    """
    assert len(tokens) == len(probs), "Tokens and token probabilities does not match."
    
    end_pointer, end_pos = 0, []
    for t in tokens:
        end_pointer += len(t)
        end_pos.append(end_pointer)
    
    assert position[1] in end_pos, "- {}\n- {}\n- {}\n- {}\n- {}\n".format(position, tokens, probs, entity, end_pos)
    last_index = end_pos.index(position[1])
    indexes = [last_index]
    total_length = len(tokens[last_index])
    
    while total_length < (position[1] - position[0]):
        last_index -= 1
        assert last_index >= 0
        indexes.append(last_index)
        total_length += len(tokens[last_index])
    
    indexes.reverse()
    
    generated = ''.join([tokens[i] for i in indexes])
    assert entity in generated, 'entity: {}; span: {}'.format(entity, generated)
    
    prob = 1.0
    for i in indexes:
        prob *= probs[i]
    return prob


def get_cmlm_probability(generator, src_input, tgt_input, position, entity):
    outputs = generator.generate(src_input, tgt_input=tgt_input)
    init_input, tokens, token_probs = outputs
    
    probs = []
    for p, tok, tokp, e in zip(position, tokens, token_probs, entity):
        probs.append(get_probability(p, tok, tokp, e).item())
    
    return probs


def get_probability_parallel(generator, src_input, tgt_input, position, entity, mask_filling=False):
    """Get entities probability in parallel decoding.

    Args:
        generator: model
        args*: outputs from prepare_cmlm_inputs()

    """
    token_probs, target = generator.encode_decode(src_input, tgt_input=tgt_input, mask_filling=mask_filling)

    probs = []
    for p, tok, tokp, e in zip(position, target, token_probs, entity):
        if mask_filling:
            assert tok[0].item() == 0
            tok, tokp = tok[1:], tokp[1:]
        
        tok_ = []
        for t in tok:
            if t.item() == 1:
                tok_.append("<pad>")
            else:
                tok_.append(generator.decode_func(t.unsqueeze(0)))
        probs.append(get_probability(p, tok_, tokp, e).item())
    
    return probs


def get_prior_probability(generator, src_input, tgt_input, position, entity):
    """Tokenize input with a special <mask> token."""
    assert len(src_input) == len(tgt_input), "source & target length should match."
    decoder_output = generator.mask_filling(src_input, tgt_input)
    init_input, tokens, token_probs = decoder_output
    
    probs = []
    for p, tok, tokp, e in zip(position, tokens, token_probs, entity):
        probs.append(get_probability(p, tok, tokp, e).item())
    return probs


def prepare_clm_inputs(source, target, ent_parts=None):
    """For Conditional Language Model. For XSum BART only."""
    if ent_parts is None:
        ent_parts = nlp(target).to_json()['ents']
    
    entities, positions = [], []
    inputs, targets = [], []

    for e in ent_parts:
        inputs.append(source)
        targets.append(target)
        positions.append((e['start'], e['end']))
        entities.append(target[e['start']: e['end']])

    return inputs, targets, positions, entities


def prepare_mlm_inputs(source, target, ent_parts=None):
    """For Masked Language Model. For BART only."""
    if ent_parts is None:
        ent_parts = nlp(target).to_json()['ents']
    
    inputs, targets = [], []
    positions, entities = [], []

    for e in ent_parts:
        inputs.append(target[0: e['start']] + '<mask>' + target[e['end']:])
        targets.append(target)
        entities.append(target[e['start']: e['end']])
        positions.append((e['start'], e['end']))
    
    return inputs, targets, positions, entities


def prepare_cmlm_inputs(source, target, ent_parts=None):
    """For Conditional Masked Language Model."""
    if ent_parts is None:
        ent_parts = nlp(target).to_json()['ents']
    
    inputs, targets = [], []
    positions, entities = [], []

    for e in ent_parts:
        masked_hypothesis = target[0: e['start']] + '###' + target[e['end']:]
        masked_hypothesis = '<s> ' + masked_hypothesis + ' <\s> ' + source
        inputs.append(masked_hypothesis)
        targets.append('<s> ' + target)
        
        entities.append(target[e['start']: e['end']])
        positions.append((e['start'] + 4, e['end'] + 4))

    return inputs, targets, positions, entities


def prepare_cmlm_ent_inputs(source, target, ent_parts=None):
    """For Entity Conditional Masked Language Model."""
    if ent_parts is None:
        ent_parts = nlp(target).to_json()['ents']
    
    inputs, targets, entities = [], [], []

    for e in ent_parts:
        masked_hypothesis = target[0: e['start']] + '###' + target[e['end']:]
        masked_hypothesis = '<s> ' + masked_hypothesis + ' <\s> ' + source
        inputs.append(masked_hypothesis)
        targets.append('<s> ' + target[e['start']: e['end']])
        
        entities.append(target[e['start']: e['end']])

    return inputs, targets, entities


def process_document(raw_doc):
    TRIVIAL_SENTS = [
        'Share this with',
        'Copy this link',
        'These are external links and will open in a new window',
    ]
    
    raw_doc = raw_doc.strip()
    raw_doc_sents = raw_doc.split('\n')
    
    start_signal = False
    filtered_sentences = []
    for s in raw_doc_sents: 
        if start_signal:
            filtered_sentences.append(s)
        elif len(s.split()) > 1 and s not in TRIVIAL_SENTS:
            start_signal = True
            filtered_sentences.append(s)
            
    return ' '.join(filtered_sentences)


def read_document(bbcid, folder):
    file_path = folder + '{}.document'.format(bbcid)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return process_document(f.read())
    else:
        return None
