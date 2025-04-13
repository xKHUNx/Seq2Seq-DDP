import os
import re
import json
import argparse

from constant import *


def extract_structured_text(split, structure_type, data_dir, max_edu=500):
    """Convert original json files into Seq2Seq-DDP structured text.

    Args:
        split (str): Choose from 'train', 'dev', 'test'.
        structure_type (str): Choose from 'natural', 'augmented'.
        data_dir (str): Directory containing the dataset files.
        max_edu (int, optional): Defaults to 500.
    """
    
    # special tokens
    BEGIN_EDU_TOKEN = '['
    END_EDU_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_TOKEN = '='

    assert structure_type in ['natural', 'augmented', 'labelmasked'], f"Structure type: {structure_type} unknown"
    
    trainset = os.path.join(data_dir, "train.json")
    devset = os.path.join(data_dir, "dev.json")
    testset = os.path.join(data_dir, "test.json")
    
    splitf = {'train': trainset, 'dev': devset, 'test': testset}

    train_dataset = []
    
    # Read input file
    with open(splitf[split], 'r') as inf:
        docs = json.load(inf)
        
    for _, l in enumerate(docs):
        dial = l
                        
        input_text = []
        output_struct = []
        train_dataset_dict = {}
        train_dataset_dict['id'] = dial['id']
        text_length = len(dial['edus'])
        
        # start parsing a doc
        if not text_length > max_edu:
            for j, edu in enumerate(dial['edus']):
                if structure_type == 'augmented': #example: [ Dave: I can trade wheat or clay | edu1 | Elaboration = edu0 ]
                    if '[' in edu['text'] or ']' in edu['text']:
                        text2 = edu['text'].replace('[', '').replace(']', '').replace('|', '') # remove all []| symbols in the text as they make confusions with augmented strucutre
                    else:
                        text2 = edu['text']
                    spktext = f"{edu['speaker']}: {text2}"
                    input_text.append(f"{BEGIN_EDU_TOKEN} {spktext} {END_EDU_TOKEN}")
                    output_begin = f"{BEGIN_EDU_TOKEN} {spktext} {SEPARATOR_TOKEN} edu{j} {SEPARATOR_TOKEN} "
                    if j == 0:
                        rel = 'root = edu0'
                        output_begin += f"{rel} {END_EDU_TOKEN}"
                    else:
                        for k, rel in enumerate(dial['relations']):
                            eduy = int(rel['y'])
                            if eduy == j: 
                                edux = int(rel['x'])
                                rel = rel['type']
                                output_begin += f"{rel} {RELATION_TOKEN} edu{edux} "
                        output_begin += f"{END_EDU_TOKEN}"
                    output_struct.append(output_begin)
                    
                elif structure_type == 'natural': #example: [edu1] is Elaboration of [edu0];
                    spktext = f"[edu{j}] {edu['speaker']}: {edu['text']}"
                    input_text.append(spktext)
                    output_begin = f"[edu{j}] is "
                    if j == 0:
                        rel = 'root'
                        output_begin += rel
                    else:
                        for k, rel in enumerate(dial['relations']):
                            eduy = int(rel['y'])
                            if eduy == j: 
                                edux = int(rel['x'])
                                rel = rel['type']
                                output_begin += f"{rel} of [edu{edux}] "
                        output_begin = output_begin[:-1] 
                    output_struct.append(output_begin)

                elif structure_type == 'labelmasked': #example: [edu1] is rel4 of [edu0];
                    if '[' in edu['text'] or ']' in edu['text']:
                        text2 = edu['text'].replace('[', '').replace(']', '').replace('|', '') # remove all []| symbols in the text
                    else:
                        text2 = edu['text']
                    spktext = f"[edu{j}] {edu['speaker']}: {text2}"
                    input_text.append(spktext)
                    output_begin = f"[edu{j}] is "
                    if j == 0:
                        rel = 'root'
                        output_begin += rel
                    else:
                        for _, rel in enumerate(dial['relations']):
                            eduy = int(rel['y'])
                            if eduy == j: 
                                edux = int(rel['x'])
                                rel = rel['type']
                                maskedrel = MASKLABEL[rel]
                                output_begin += f"{maskedrel} of [edu{edux}] "
                        output_begin = output_begin[:-1]
                    output_struct.append(output_begin)
            
            input_dial = " ".join(input_text)
            if structure_type == 'augmented':
                output_dial = " ".join(output_struct)    
            else:
                output_dial = "; ".join(output_struct)
            train_dataset_dict['dialogue'] = input_dial
            train_dataset_dict['structure'] = output_dial
            train_dataset.append(train_dataset_dict)
        
    outfname = os.path.join(data_dir, f"{structure_type}_{split}.json")
    with open(outfname, "w") as outf:
        for dict in train_dataset:
            string = json.dumps(dict)
            outf.write(string+'\n')

def extract_transition_based_text(split, structure_type, data_dir):
    """Generate transition-based data set.

    Args:
        split (str): Choose from 'train', 'dev', 'test'.
        structure_type (str): Choose from 'natural2', 'focus'.
        data_dir (str): Directory containing the dataset files.
    """
    assert structure_type in ['natural2', 'focus'], f"Transition-based structure type: {structure_type} unknown" 
    
    with open(f"{data_dir}/natural_{split}.json", 'r') as f:
        lines = f.readlines()
    outf = open(f'{data_dir}/{structure_type}_{split}.json', 'w')
    
    diffs = []
    for line in lines:
        dialogue = json.loads(line)
        id = dialogue['id']
        if id in ['s1-league1-game3_3', 's2-league1-game1_19']:
            continue

        # New EDU parsing logic
        edus = []
        dialogue_text = dialogue['dialogue']
        # Find all [eduX] patterns
        edu_markers = re.finditer(r'\[edu\d+\]', dialogue_text)
        
        # Get the positions of all EDU markers
        positions = [(m.start(), m.end()) for m in edu_markers]
        
        # Extract text between EDU markers
        for i in range(len(positions)):
            start = positions[i][1]  # End of current EDU marker
            end = positions[i+1][0] if i < len(positions)-1 else len(dialogue_text)
            edu_text = dialogue_text[start:end].strip()
            edus.append(f"edu{i}")  # Add EDU marker
            edus.append(edu_text)   # Add EDU text
            
        relations = re.split(';', dialogue['structure'])
        relations = [relation.strip() for relation in relations]
        
        # for idx, edu in enumerate(edus):
        #   print(idx, edu)
        # print(len(edus), len(relations))
        assert len(edus) == 2*len(relations), f"{id}: {edus}"

        diff = 0
        for relation in relations:
            _relation = re.split('[\[\]]', relation)
            _relation = [_r.strip() for _r in _relation if _r.strip()]
            _relation = [_r for _r in _relation if 'edu' in _r]
            if len(_relation) > 1:
                _relation_n = [int(''.join(re.findall('\d', _r))) for _r in _relation]
                for _i in range(len(_relation_n)-1):
                    for _j in range(_i+1, len(_relation_n)):
                        if diff < abs(_relation_n[_i] - _relation_n[_j]):
                            diff = abs(_relation_n[_i] - _relation_n[_j])        
        diffs.append(diff)
                
        if structure_type == 'focus': 
            _dialogues = ['[{}] {}'.format(edus[0], edus[1])]
            for i in range(len(relations)):
                _structure = re.split('is', relations[i])
                assert len(_structure) == 2
                _structure = [_s.strip() for _s in _structure if _s.strip()]
                if len(_structure) > 1:
                    assert len(_structure) == 2
                else:
                    assert len(_structure) == 1
                if len(_structure) > 1:
                    _structure = '{}'.format(_structure[1])
                else:
                    _structure = ' '

                x = {'id': id + '_{:0>2d}'.format(i),
                    'dialogue': ''.join(_dialogues[-18:-1] + [' **'] + _dialogues[-1:]).strip(),
                    'structure': _structure
                    }
                x = json.dumps(x) + '\n'
                outf.write(x)

                if i < len(relations) - 1:
                    _structure = re.split('is', relations[i])
                    assert len(_structure) == 2
                    _structure = [_s.strip() for _s in _structure if _s.strip()]
                    if len(_structure) > 1:
                        assert len(_structure) == 2
                    else:
                        assert len(_structure) == 1
                    if len(_structure) > 1:
                        _dialogues[-1] += ' | {};'.format(_structure[1])
                    else:
                        _dialogues[-1] += ' | ;'
                    _dialogues.append(' [{}] {}'.format(edus[(i+1)*2], edus[(i+1)*2+1]))
        
        elif structure_type == 'natural2':
            _dialogues = ['[{}] [{}] is'.format(edus[0], edus[1])]
            for i in range(len(relations)):
                _structure = re.split('is', relations[i])
                assert len(_structure) == 2
                _structure = [_s.strip() for _s in _structure if _s.strip()]
                if len(_structure) > 1:
                    assert len(_structure) == 2
                else:
                    assert len(_structure) == 1
                if len(_structure) > 1:
                    _structure = '{}'.format(_structure[1])
                else:
                    _structure = ' '

                x = {'id': id + '_{:0>2d}'.format(i),
                'dialogue': ''.join(_dialogues[-18:]).strip(),
                'structure': _structure
                }
                x = json.dumps(x) + '\n'
                outf.write(x)

                if i < len(relations) - 1:
                    _structure = re.split('is', relations[i])
                    assert len(_structure) == 2
                    _structure = [_s.strip() for _s in _structure if _s.strip()]
                    if len(_structure) > 1:
                        assert len(_structure) == 2
                    else:
                        assert len(_structure) == 1
                    if len(_structure) > 1:
                        _dialogues[-1] += ' {};'.format(_structure[1])
                    else:
                        _dialogues[-1] += ' ;'
                    _dialogues.append(' [{}] [{}] is'.format(edus[(i+1)*2], edus[(i+1)*2+1]))
        
    outf.close()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--split", type=str, help="train, dev, test")
    parser.add_argument("--structure_type", type=str, help="end2end: 'natural', 'augmented', 'labelmasked' | transition-based: 'focus', 'natural2'.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing the dataset files")
    args = parser.parse_args()
    
    split = args.split
    structure_type = args.structure_type
    data_dir = args.data_dir

    # Check structure type and call appropriate function
    if structure_type in ['natural', 'augmented', 'labelmasked']:
        extract_structured_text(split, structure_type, data_dir)
    elif structure_type in ['natural2', 'focus']:
        extract_transition_based_text(split, structure_type, data_dir)
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")
 