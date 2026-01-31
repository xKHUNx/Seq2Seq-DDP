import os
import re
import json
import argparse

from constant import *


def extract_structured_text(split, structure_type, data_dir, max_edu=500, window_size=10, stride=3):
    """Convert original json files into Seq2Seq-DDP structured text with sliding window.

    Args:
        split (str): Choose from 'train', 'dev', 'test'.
        structure_type (str): Choose from 'natural', 'augmented'.
        data_dir (str): Directory containing the dataset files.
        max_edu (int, optional): Defaults to 500.
        window_size (int): Size of sliding window. Defaults to 10.
        stride (int): Stride for sliding window. Defaults to 3.
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
        text_length = len(dial['edus'])
        
        # Skip if dialogue is too long
        if text_length > max_edu:
            continue
            
        # Determine windows for processing
        windows = []
        if window_size == -1 and stride == -1:
            if text_length >= 2:
                windows.append((0, text_length))
        else:
            for window_start in range(0, max(1, text_length - window_size + 1), stride):
                window_end = min(window_start + window_size, text_length)
                if window_end - window_start >= 2:
                    windows.append((window_start, window_end))

        for window_start, window_end in windows:
            input_text = []
            output_struct = []
            train_dataset_dict = {}
            if window_size == -1 and stride == -1:
                train_dataset_dict['id'] = dial['id']
            else:
                train_dataset_dict['id'] = f"{dial['id']}_window_{window_start}_{window_end}"
            
            # Get relations that fall within this window
            window_relations = []
            for rel in dial['relations']:
                edux = int(rel['x'])
                eduy = int(rel['y'])
                # Keep relation if both x and y are within window
                if window_start <= edux < window_end and window_start <= eduy < window_end:
                    # Adjust indices relative to window start
                    adjusted_rel = {
                        'x': str(edux - window_start),
                        'y': str(eduy - window_start),
                        'type': rel['type']
                    }
                    window_relations.append(adjusted_rel)
            
            # Process EDUs in the window
            for j in range(window_start, window_end):
                window_j = j - window_start  # Adjust index relative to window
                edu = dial['edus'][j]
                
                if structure_type == 'augmented': #example: [ Dave: I can trade wheat or clay | edu1 | Elaboration = edu0 ]
                    if '[' in edu['text'] or ']' in edu['text']:
                        text2 = edu['text'].replace('[', '').replace(']', '').replace('|', '') # remove all []| symbols in the text as they make confusions with augmented strucutre
                    else:
                        text2 = edu['text']
                    spktext = f"{edu['speaker']}: {text2}"
                    input_text.append(f"{BEGIN_EDU_TOKEN} {spktext} {END_EDU_TOKEN}")
                    output_begin = f"{BEGIN_EDU_TOKEN} {spktext} {SEPARATOR_TOKEN} edu{window_j} {SEPARATOR_TOKEN} "
                    
                    if window_j == 0:
                        rel = 'root = edu0'
                        output_begin += f"{rel} {END_EDU_TOKEN}"
                    else:
                        relations_found = False
                        for k, rel in enumerate(window_relations):
                            eduy = int(rel['y'])
                            if eduy == window_j: 
                                edux = int(rel['x'])
                                rel_type = rel['type']
                                output_begin += f"{rel_type} {RELATION_TOKEN} edu{edux} "
                                relations_found = True
                        if not relations_found:
                            # If no relation found, make it root
                            output_begin += f"root {RELATION_TOKEN} edu0 "
                        output_begin += f"{END_EDU_TOKEN}"
                    output_struct.append(output_begin)
                    
                elif structure_type == 'natural': #example: [edu1] is Elaboration of [edu0];
                    spktext = f"[edu{window_j}] {edu['speaker']}: {edu['text']}"
                    input_text.append(spktext)
                    output_begin = f"[edu{window_j}] is "
                    
                    if window_j == 0:
                        rel = 'root'
                        output_begin += rel
                    else:
                        relations_found = False
                        for k, rel in enumerate(window_relations):
                            eduy = int(rel['y'])
                            if eduy == window_j: 
                                edux = int(rel['x'])
                                rel_type = rel['type']
                                output_begin += f"{rel_type} of [edu{edux}] "
                                relations_found = True
                        if not relations_found:
                            # If no relation found, make it root
                            output_begin += "root "
                        output_begin = output_begin[:-1] 
                    output_struct.append(output_begin)

                elif structure_type == 'labelmasked': #example: [edu1] is rel4 of [edu0];
                    if '[' in edu['text'] or ']' in edu['text']:
                        text2 = edu['text'].replace('[', '').replace(']', '').replace('|', '') # remove all []| symbols in the text
                    else:
                        text2 = edu['text']
                    spktext = f"[edu{window_j}] {edu['speaker']}: {text2}"
                    input_text.append(spktext)
                    output_begin = f"[edu{window_j}] is "
                    
                    if window_j == 0:
                        rel = 'root'
                        output_begin += rel
                    else:
                        relations_found = False
                        for _, rel in enumerate(window_relations):
                            eduy = int(rel['y'])
                            if eduy == window_j: 
                                edux = int(rel['x'])
                                rel_type = rel['type']
                                maskedrel = MASKLABEL[rel_type]
                                output_begin += f"{maskedrel} of [edu{edux}] "
                                relations_found = True
                        if not relations_found:
                            # If no relation found, make it root
                            output_begin += "root "
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

def extract_transition_based_text(split, structure_type, data_dir, window_size=10, stride=3):
    """Generate transition-based data set with sliding window.

    Args:
        split (str): Choose from 'train', 'dev', 'test'.
        structure_type (str): Choose from 'natural2', 'focus'.
        data_dir (str): Directory containing the dataset files.
        window_size (int): Size of sliding window. Defaults to 10.
        stride (int): Stride for sliding window. Defaults to 3.
    """
    assert structure_type in ['natural2', 'focus'], f"Transition-based structure type: {structure_type} unknown" 
    
    with open(f"{data_dir}/natural_{split}.json", 'r') as f:
        lines = f.readlines()
    outf = open(f'{data_dir}/{structure_type}_{split}.json', 'w')
    
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
        
        assert len(edus) == 2*len(relations), f"{id}: {edus}"
        
        num_edus = len(relations)
        
        # Apply sliding window to relations
        windows = []
        if window_size == -1 and stride == -1:
            if num_edus >= 2:
                windows.append((0, num_edus))
        else:
            for window_start in range(0, max(1, num_edus - window_size + 1), stride):
                window_end = min(window_start + window_size, num_edus)
                if window_end - window_start >= 2:
                    windows.append((window_start, window_end))

        for window_start, window_end in windows:
            window_relations = relations[window_start:window_end]
            window_edus = edus[window_start*2:window_end*2]  # Each EDU has 2 elements (marker + text)
            
            # Adjust EDU indices in relations to be relative to window
            adjusted_relations = []
            for rel in window_relations:
                # Parse and adjust EDU indices in the relation text
                adjusted_rel = rel
                # Find all edu references and adjust them
                edu_pattern = r'edu(\d+)'
                def adjust_edu_ref(match):
                    edu_idx = int(match.group(1))
                    if window_start <= edu_idx < window_end:
                        return f'edu{edu_idx - window_start}'
                    else:
                        return match.group(0)  # Keep original if outside window
                
                adjusted_rel = re.sub(edu_pattern, adjust_edu_ref, adjusted_rel)
                adjusted_relations.append(adjusted_rel)
            
            if window_size == -1 and stride == -1:
                window_id = id
            else:
                window_id = f"{id}_window_{window_start}_{window_end}"
            
            if structure_type == 'focus': 
                _dialogues = ['[{}] {}'.format(window_edus[0], window_edus[1])]
                for i in range(len(adjusted_relations)):
                    _structure = re.split('is', adjusted_relations[i])
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

                    x = {'id': window_id + '_{:0>2d}'.format(i),
                        'dialogue': ''.join(_dialogues[-18:-1] + [' **'] + _dialogues[-1:]).strip(),
                        'structure': _structure
                        }
                    x = json.dumps(x) + '\n'
                    outf.write(x)

                    if i < len(adjusted_relations) - 1:
                        _structure = re.split('is', adjusted_relations[i])
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
                        _dialogues.append(' [{}] {}'.format(window_edus[(i+1)*2], window_edus[(i+1)*2+1]))
            
            elif structure_type == 'natural2':
                _dialogues = ['[{}] [{}] is'.format(window_edus[0], window_edus[1])]
                for i in range(len(adjusted_relations)):
                    _structure = re.split('is', adjusted_relations[i])
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

                    x = {'id': window_id + '_{:0>2d}'.format(i),
                    'dialogue': ''.join(_dialogues[-18:]).strip(),
                    'structure': _structure
                    }
                    x = json.dumps(x) + '\n'
                    outf.write(x)

                    if i < len(adjusted_relations) - 1:
                        _structure = re.split('is', adjusted_relations[i])
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
                        _dialogues.append(' [{}] [{}] is'.format(window_edus[(i+1)*2], window_edus[(i+1)*2+1]))
        
    outf.close()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--split", type=str, help="train, dev, test")
    parser.add_argument("--structure_type", type=str, help="end2end: 'natural', 'augmented', 'labelmasked' | transition-based: 'focus', 'natural2'.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the dataset files")
    parser.add_argument("--window_size", type=int, default=10, help="Size of sliding window (default: 10)")
    parser.add_argument("--stride", type=int, default=3, help="Stride for sliding window (default: 3)")
    args = parser.parse_args()
    
    split = args.split
    structure_type = args.structure_type
    data_dir = args.data_dir
    window_size = args.window_size
    stride = args.stride

    # Check structure type and call appropriate function
    if structure_type in ['natural', 'augmented', 'labelmasked']:
        extract_structured_text(split, structure_type, data_dir, window_size=window_size, stride=stride)
    elif structure_type in ['natural2', 'focus']:
        extract_transition_based_text(split, structure_type, data_dir, window_size=window_size, stride=stride)
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")