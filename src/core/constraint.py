from rdkit import Chem
from src.utils.nmr_split import predict_h_split
from typing import List, Dict, Any, Optional


def satisfy_allowed_elements(mol: Chem.Mol, allowed_elements: Optional[set]) -> bool:
    """
    Check if a molecule contains only allowed elements.
    """
    if allowed_elements is None:
        return True
    elements = set([atoms.GetSymbol() for atoms in mol.GetAtoms()] + ['H'])
    return elements.issubset(set(allowed_elements))


def satisfy_elements(mol: Chem.Mol, elements: List[str]) -> bool:
    """
    Check if a molecule contains exactly the target elements.
    """
    return set([atoms.GetSymbol() for atoms in mol.GetAtoms()] + ['H']) == set(elements)


def satisfy_formula(mol: Chem.Mol, formula: str) -> bool:
    """
    Check if a molecule matches the target formula.
    """
    return Chem.rdMolDescriptors.CalcMolFormula(mol) == formula


def satisfy_split(mol: Chem.Mol, H_split: List[str]) -> bool:
    """
    Check if a molecule satisfies the target hydrogen splitting.
    """
    n_split = {}
    for item in H_split:
        if item in ['m', 'br']:
            continue
        else:
            if item == 'hept':
                mode = 'hept'
            else:
                for s in item:
                    assert s in ['s', 'd', 't', 'q', 'p', 'h'], f"Invalid split mode: {item[0]}"
                mode = ''.join(sorted(item))
            if mode not in n_split:
                n_split[mode] = 0
            n_split[mode] += 1
            
    split = predict_h_split(mol)
    n_split = {}
    for item in split.values():
        for s in item[0]:
            if s not in n_split:
                n_split[s] = 0
            n_split[s] += item[1]
    
    for mode in n_split:
        if mode not in n_split or n_split[mode] < n_split[mode]:
            return False
    return True
        

def satisfy_constraints(mol: Chem.Mol, constraints: Dict[str, Any]) -> bool:
    """
    Check if a molecule satisfies all given constraints.
    """
    if mol is None:
        return False
    
    constraints_check_func = {
        "allowed_elements": satisfy_allowed_elements,
        "elements": satisfy_elements,
        "formula": satisfy_formula,
        "H_split": satisfy_split,
    }
    for key in constraints:
        if key not in constraints_check_func:
            raise ValueError(f"Unknown constraint key: {key}")
        value = constraints[key]
        if not constraints_check_func[key](mol, value):
            return False
    return True
