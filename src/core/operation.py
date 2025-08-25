# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
rdBase.DisableLog('rdApp.error')

from collections import defaultdict
from typing import List, Tuple, Optional

from src.utils.chem_tools import has_radical, get_canonical_smiles_from_mol
from src.utils.nmrnet import get_equi_class


def get_complement_cut_type(cut_type: tuple) -> tuple:
    """
    Get the complement of a cut type.
    """
    # cut_non_ring: ((atom1, atom2), (bond1)) -> ((atom2, atom1), (bond1))
    # cut_ring: ((atom1, atom2, atom3, atom4), (bond1, bond2)) -> ((atom2, atom1, atom4, atom3), (bond1, bond2))
    if len(cut_type[1]) == 1:
        return ((cut_type[0][1], cut_type[0][0]), cut_type[1])
    else:
        assert len(cut_type[1]) == 2
        return ((cut_type[0][1], cut_type[0][0], cut_type[0][3], cut_type[0][2]), cut_type[1])


def cut_non_ring(mol: Chem.Mol) -> List[Tuple[Chem.Mol, Tuple]]:
    """
    Cut non-ring bonds in a molecule.
    """
    fragments_mol = []
    bond_map = {Chem.BondType.SINGLE: '-', Chem.BondType.DOUBLE: '=', Chem.BondType.TRIPLE: '#'}
    
    for bis in mol.GetSubstructMatches(Chem.MolFromSmarts('[*]~;!@[*]')):
        bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
        fragments = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])
        
        atom_symbol = [mol.GetAtomWithIdx(bis[i]).GetSymbol() for i in range(2)]
        bond_type = bond_map[mol.GetBondBetweenAtoms(bis[0], bis[1]).GetBondType()]
        fragments = list(Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=False))
        if len(fragments) != 2:
            continue
        if all(frag.GetNumAtoms() > 2 for frag in fragments) or all(symbol not in ['H', 'F', 'Cl', 'Br', 'I'] for symbol in atom_symbol):
            cut_type = ((atom_symbol[0], atom_symbol[1]), (bond_type))
            fragments_mol += [(fragments[0], cut_type), (fragments[1], get_complement_cut_type(cut_type))]
        elif fragments[1].GetNumAtoms() == 2 and bond_type == '-' and atom_symbol[1] in ['H', 'F', 'Cl', 'Br', 'I']:
            cut_type = ((atom_symbol[0], 'X'), (bond_type))
            fragments_mol += [(fragments[0], cut_type)]
        elif fragments[0].GetNumAtoms() == 2 and bond_type == '-' and atom_symbol[0] in ['H', 'F', 'Cl', 'Br', 'I']:
            cut_type = ((atom_symbol[1], 'X'), (bond_type))
            fragments_mol += [(fragments[1], cut_type)]
        else:
            print("Warning: Unexpected encountered.")
            print(Chem.MolToSmiles(fragments[0]), Chem.MolToSmiles(fragments[1]))

    return fragments_mol


def cut_ring(mol: Chem.Mol) -> List[Tuple[Chem.Mol, Tuple]]:
    """
    Cut ring bonds in a molecule.
    """
    fragments_mol = []
    bond_map = {Chem.BondType.SINGLE: '-', Chem.BondType.DOUBLE: '=', Chem.BondType.TRIPLE: '#'}
    structure_bond = {'[R]!:@[R]!:@[R]!:@[R]': (0,1,2,3), '[R]!:@[R]!:@[R]': (0,1,1,2)} # Filter aromatic bonds
    
    for structure in structure_bond:
        bond = structure_bond[structure]
        for bis in mol.GetSubstructMatches(Chem.MolFromSmarts(structure)):
            bis = ((bis[bond[0]], bis[bond[1]]), (bis[bond[2]], bis[bond[3]]),)
            bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]
            fragments = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (2, 2)])
            
            atom_symbol = [mol.GetAtomWithIdx(bis[i][j]).GetSymbol() for i in range(2) for j in range(2)]
            bond_type = [bond_map[mol.GetBondBetweenAtoms(x, y).GetBondType()] for x, y in bis]
            try:
                fragments = Chem.GetMolFrags(fragments, asMols=True, sanitizeFrags=False)
                if len(fragments) != 2:
                    continue
                cut_type = ((atom_symbol[0], atom_symbol[1], atom_symbol[2], atom_symbol[3]), (bond_type[0], bond_type[1]))
                fragments_mol += [(fragments[0], cut_type), (fragments[1], get_complement_cut_type(cut_type))]
            except ValueError:
                continue

    return fragments_mol


def stitch_fragment(fragment1: Chem.Mol, fragment2: Chem.Mol, bond_type: Tuple) -> Chem.Mol:
    """
    Stitch two fragments together.
    """
    if len(bond_type) == 1:
        rxn = AllChem.ReactionFromSmarts(f'[*:1]~[1*].[1*]~[*:2]>>[*:1]{bond_type[0]}[*:2]')
        mol_gen = rxn.RunReactants((fragment1, fragment2))[0][0]
    else:
        assert len(bond_type) == 2
        rxn1 = AllChem.ReactionFromSmarts(f'[*:1]~[1*].[1*]~[*:2]>>[*:1]{bond_type[0]}[*:2]')
        rxn2 = AllChem.ReactionFromSmarts(f'([*:1]~[2*].[2*]~[*:2])>>[*:1]{bond_type[1]}[*:2]')
        intermediate = rxn1.RunReactants((fragment1, fragment2))[0][0]
        intermediate.UpdatePropertyCache(strict=False)
        mol_gen = rxn2.RunReactants((intermediate,))[0][0]
    return mol_gen


def stitch_fragment_to_smiles(input: Tuple[Chem.Mol, Chem.Mol, str | List[str]], max_cycle_length: Optional[int] = None, invalid_patterns: List[str] = []) -> str | None:
    """
    Stitch two fragments and return the SMILES representation.
    """
    frag1, frag2, bond_type = input
    mol = stitch_fragment(frag1, frag2, bond_type)
    
    try:
        Chem.SanitizeMol(mol)
    except:
        return None
    
    if not mol_ok(mol, max_cycle_length=max_cycle_length, invalid_patterns=invalid_patterns):
        return None
        
    smiles = get_canonical_smiles_from_mol(mol, remove_hs=True)
    return smiles


def replace_halogens(mol: Chem.Mol, equi_class: Optional[List[int]] = None, optional_elements: List[str] = ['F', 'Cl', 'Br', 'I']) -> List[str]:
    """
    Replace halogens in a molecule.
    """
    equi_class = equi_class or get_equi_class(mol).tolist()
    halogens = ['F','Cl','Br','I']
    for element in optional_elements:
        assert element in halogens, 'element must be one of F, Cl, Br, I'
    
    halogan_atom_equi_class = defaultdict(list)
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() in halogens and len(atom.GetNeighbors()) == 1 and atom.GetBonds()[0].GetBondType() == Chem.BondType.SINGLE:
            halogan_atom_equi_class[equi_class[i]].append(i)
    halogan_equi_atoms = list(halogan_atom_equi_class.values())
    
    if len(halogan_equi_atoms) == 0:
        return []
    else:
        new_mol_list = []
        for i, equi_atoms in enumerate(halogan_equi_atoms):
            origin_element = mol.GetAtomWithIdx(equi_atoms[0]).GetSymbol()
            for target in optional_elements:
                if target == origin_element:
                    continue
                new_mol = Chem.Mol(mol)
                for j in equi_atoms:
                    atom = new_mol.GetAtomWithIdx(j)
                    atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(target))
                try:
                    new_mol_list.append(new_mol)
                except:
                    continue
        return new_mol_list
    
    
def check_validity(mol: Chem.Mol | str) -> bool:
    """
    Check the validity of a molecule.
    """
    try:
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
        Chem.SanitizeMol(mol)
    except ValueError:
        return False
    
    return not has_radical(mol)


def mol_ok(mol: Chem.Mol, max_cycle_length: Optional[int] = None, invalid_patterns: List[str] = []) -> bool:
    """
    Check if a molecule satisfies certain conditions.
    """
    for pattern in invalid_patterns:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
            return False
    
    if max_cycle_length:
        cycle_list = mol.GetRingInfo().AtomRings()
        if cycle_list and max([len(j) for j in cycle_list]) > max_cycle_length:
            return False

    return True


if __name__=="__main__":
    mol1 = Chem.MolFromSmiles('CCO')
    mol1 = Chem.AddHs(mol1)
    result = cut_non_ring(mol1)
    for frag, atom_symbol in result:
        print(Chem.MolToSmiles(frag), atom_symbol)

