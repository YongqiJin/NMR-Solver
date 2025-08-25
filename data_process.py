from src.utils.nmr_parser import parse_h_raw, parse_h_split, parse_h_shifts, parse_c_shifts
from src.utils.data_tools import write_lmdb


def process_nmr_txt(txt_path, save_path=None):
    
    input_data = open(txt_path, 'r').read()

    result = []
    for id, item in enumerate(input_data.split('\n\n')):
        item = item.strip()
        if len(item.split('\n')) == 3:
            try:
                item1, item2, item3 = item.split('\n')[0], item.split('\n')[1], item.split('\n')[2]
            except:
                raise 
        elif len(item.split('\n')) == 4:
            try:
                item0, item1, item2, item3 = item.split('\n')[0], item.split('\n')[1], item.split('\n')[2], item.split('\n')[3]
            except:
                raise
        else:
            print(f"{id} failed: {item}")
            continue
        try:
            candidates = item0.split(', ') if len(item.split('\n')) == 4 else []
            smi = item1
            h_raw = parse_h_raw(item2)
            h_split = parse_h_split(h_raw)
            h_shifts = parse_h_shifts(h_raw)
            c_shifts = parse_c_shifts(item3)
            result.append({
                'candidates': candidates,
                'smiles': smi,
                'nmr_text': '\n'.join([item2, item3]),
                'H_raw': h_raw,
                'H_split': h_split,
                'H_shifts': h_shifts,
                'C_shifts': c_shifts,
            })
                
        except:
            print(f"{id} failed")
            continue
    
    print('num:', len(result))
    print('example:', {0: result[0]})
    
    if save_path is None:
        save_path = txt_path.replace(".txt", ".lmdb")
    
    write_lmdb(result, save_path)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="./data/demo/test.txt")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()
    process_nmr_txt(txt_path=args.input, save_path=args.output)
