from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import Draw
import numpy as np
from typing import List, Tuple, Union
import os
import re

from PIL import Image

from collections import defaultdict
from IPython.display import SVG, display


def get_canonical_smiles_from_mol(mol: Chem.Mol, remove_hs: bool = True) -> str:
    """
    Generates a canonical SMILES string from a molecule.

    Returns:
        str: Canonical SMILES string.
    """
    try:
        mol_copy = Chem.Mol(mol)
        # remove AtomMapNum
        for atom in mol_copy.GetAtoms():
            atom.SetAtomMapNum(0)
        if remove_hs:
            mol_copy = Chem.RemoveHs(mol_copy)
        return Chem.MolToSmiles(mol_copy)
    except:
        return None


def save_img(content: Union[str, Image.Image], save_path: str):
    """
    Saves an image or SVG content to a file.

    Args:
        content (Union[str, Image.Image]): SVG content as a string or a PIL Image object.
        save_path (str): Path to save the file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(content, str) and save_path.endswith(".svg"):
        with open(save_path, "w") as f:
            f.write(content)
    elif isinstance(content, Image.Image):
        content.save(save_path)
    else:
        raise ValueError("Unsupported content format for saving: {}".format(type(content)))


def generate_svg(
    mols: List[Chem.Mol],
    dos: Draw.MolDrawOptions = None,
    size: Tuple[int, int] = (300, 300),
    legends: List[str] = None,
    mols_per_row: int = None
) -> str:
    """
    Generate an SVG of multiple molecules arranged in a grid.

    Args:
        mols: List of RDKit molecule objects.
        dos: Optional MolDrawOptions object.
        size: (width, height) of each molecule drawing.
        legends: Optional list of legend strings.
        mols_per_row: How many molecules per row. Defaults to all in one row.

    Returns:
        SVG string of the grid.
    """
    if legends is None:
        legends = [''] * len(mols)
    if mols_per_row is None:
        mols_per_row = len(mols)
    if dos is None:
        dos = Draw.MolDrawOptions()

    n_rows = (len(mols) + mols_per_row - 1) // mols_per_row
    canvas_width = size[0] * mols_per_row
    canvas_height = size[1] * n_rows

    drawer = Draw.MolDraw2DSVG(canvas_width, canvas_height, size[0], size[1])
    drawer.SetDrawOptions(dos)
    drawer.DrawMolecules(mols, legends=legends)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    svg = svg.replace('<svg', '<svg style="background-color:white;"')
    pattern = r"(class='note'.*?fill=')#000000'"
    replacement = r"\1" + "#0000FF'"  # Change the annonation color to blue
    svg = re.sub(pattern, replacement, svg, flags=re.DOTALL)
    return svg


def draw_mol(mols: List, dos=None, size: Tuple = (300, 300), add_hs: bool = False, add_index: bool = False, mols_per_row: int = None, save_path: str = None, is_display: bool = False) -> str:
    """
    Draws molecules and optionally saves the SVG.

    Returns:
        str: Generated SVG content.
    """
    if len(mols) > 0 and isinstance(mols[0], str):
        mols = [Chem.MolFromSmiles(mol) for mol in mols]
    
    mols_to_plot = []
    for mol in mols:
        mol_copy = Chem.Mol(mol)
        mol_copy.RemoveAllConformers()
        for atom in mol_copy.GetAtoms():
            atom.SetAtomMapNum(0)
        mols_to_plot.append(Chem.AddHs(mol_copy) if add_hs else Chem.RemoveHs(mol_copy))
    
    if dos is None:
        dos = Draw.MolDrawOptions()
        if add_index:
            dos.addAtomIndices = True
    
    svg = generate_svg(mols_to_plot, dos=dos, size=size, mols_per_row=mols_per_row)
    if is_display:
        display(SVG(svg))
    if save_path:
        save_img(svg, save_path)
    return svg


def draw_mol_with_nmr(mol_list: List, shifts_list: List, nmr_type: List = ["H", "C"], mols_per_row: int = None, size: Tuple = (300, 300), fontscale: float = 0.6, save_path: str = None, is_display: bool = False):
    """
    Draws molecules with NMR shift annotations as SVG.

    Args:
        mol_list (list): List of RDKit molecule objects.
        shifts_list (list): List of NMR shifts.
        nmr_type (list): List of element type ('C', 'H') to annotate.
        mols_per_row (int, optional): Number of molecules per row. Defaults to None.
        size (tuple, optional): Image size. Defaults to (300, 300).
        fontscale (float, optional): Font scale for annotations. Defaults to 0.6.
        save_path (str, optional): Path to save the SVG. Defaults to None.
        show_in_notebook (bool, optional): Whether to display the SVG in a Jupyter Notebook. Defaults to False.

    Returns:
        str: Generated SVG content.
    """
    mols_to_plot = []
    for mol, shifts in zip(mol_list, shifts_list):
        for element in nmr_type:
            note_dict = defaultdict(list)
            mol_copy = Chem.Mol(mol)
            mol_copy.RemoveAllConformers()
            for i, (atom, shift) in enumerate(zip(mol_copy.GetAtoms(), shifts)):
                atom.SetAtomMapNum(0)
                if atom.GetSymbol() == element == "C":
                    note_dict[i].append(f"{shift:.1f}")
                elif atom.GetSymbol() == element == "H":
                    j = atom.GetNeighbors()[0].GetIdx()
                    note_dict[j].append(f"{shift:.2f}")
            for i, note in note_dict.items():
                mol_copy.GetAtomWithIdx(i).SetProp("atomNote", ", ".join(sorted(note, key=float)))
            mols_to_plot.append(Chem.RemoveHs(mol_copy))
    
    dos = Draw.MolDrawOptions()
    dos.setAnnotationColour((0, 0, 1))
    dos.annotationFontScale = fontscale
    
    map_dict = {'H': '1H NMR', 'C': '13C NMR'}
    legends = [map_dict[element] for element in nmr_type]
    svg = generate_svg(mols_to_plot, dos=dos, size=size, mols_per_row=mols_per_row, legends=legends * len(mol_list))
    if is_display:
        display(SVG(svg))
    if save_path:
        save_img(svg, save_path)
    return svg


def morgan_fp(mol: Chem.Mol, fp_size=2048, radius=2, to_np=True, **kwargs):
    """
    Compute Morgan fingerprint for a molecule.
    
    Args:
        mol (Chem.Mol): _description_
        fp_size (int, optional): Size of the fingerprint.
        radius (int, optional): Radius of the fingerprint.
        to_np (bool, optional): Convert the fingerprint to numpy array.
    """

    fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=fp_size, **kwargs)
    if to_np:
        fp_np = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, fp_np)
        fp = fp_np
    return fp


def tanimoto_morgan_similarity(mol1: Union[Chem.Mol, str], mol2: Union[Chem.Mol, str], **kwargs) -> float:
    """
    Compute Tanimoto similarity between two molecules using Morgan fingerprints.

    Args:
        mol1 (T.Union[Chem.Mol, str]): First molecule as RDKit molecule or SMILES string.
        mol2 (T.Union[Chem.Mol, str]): Second molecule as RDKit molecule or SMILES string.
    """
    if isinstance(mol1, str):
        mol1 = Chem.MolFromSmiles(mol1)
    if isinstance(mol2, str):
        mol2 = Chem.MolFromSmiles(mol2)
    return DataStructs.TanimotoSimilarity(morgan_fp(mol1, to_np=False, **kwargs), morgan_fp(mol2, to_np=False, **kwargs))


def get_elements_from_mol(input_data: Union[str, Chem.Mol]) -> list:
    """
    Extracts a list of elements from a molecule.
    """
    if isinstance(input_data, str):
        mol = Chem.MolFromSmiles(input_data)
    elif isinstance(input_data, Chem.Mol):
        mol = input_data
    else:
        raise ValueError("Input data must be a SMILES string or Chem.Mol object.")
    
    mol = Chem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return list(set(atoms))


def has_isotope(mol: Chem.Mol) -> bool:
    """
    Checks if a molecule contains isotopes.
    """
    for atom in mol.GetAtoms():
        if atom.GetIsotope() != 0:
            return True
    return False


def has_radical(mol: Chem.Mol) -> bool:
    """
    Checks if a molecule contains radicals.
    """
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True
    return False


def not_charged(mol: Chem.Mol) -> bool:
    """
    Checks if a molecule is charged based on the sum of formal charges of its atoms.
    """
    total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    return total_charge == 0


def remove_stereo(mol: Chem.Mol) -> Chem.Mol:
    """
    Removes stereochemistry information from a molecule.
    """
    mol_copy = Chem.Mol(mol)
    
    for atom in mol_copy.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in mol_copy.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
    
    try:
        Chem.SanitizeMol(mol_copy)
        return mol_copy
    except:
        return None

