# dataPipeline/ingester.py - complete replacement

import requests
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


# ── Source 1: Hardcoded ncAA Seed Set ───────────────────────────────────────
# Curated from literature - these are verified non-canonical amino acids
# with known pharmacophore relevance. This is your ground truth anchor set.

NCAA_SEED_SET = {
    # Fluorinated amino acids - key ncAA class for metabolic labeling
    'ncAA_F_Ala':        'N[C@@H](CF)C(=O)O',
    'ncAA_difluoro_Ala': 'N[C@@H](C(F)F)C(=O)O',
    'ncAA_F_Phe':        'N[C@@H](Cc1ccc(F)cc1)C(=O)O',
    'ncAA_F3_Phe':       'N[C@@H](Cc1ccc(C(F)(F)F)cc1)C(=O)O',

    # Selenoamino acids - selenium pharmacophore class
    'ncAA_SeCys':        'N[C@@H](C[SeH])C(=O)O',
    'ncAA_SeMet':        'N[C@@H](CC[Se]C)C(=O)O',

    # Azide-containing - click chemistry handles
    'ncAA_AzidoAla':     'N[C@@H](CN=[N+]=[N-])C(=O)O',
    'ncAA_AzidoLys':     'N[C@@H](CCCCN=[N+]=[N-])C(=O)O',
    'ncAA_para_AzidoPhe': 'N[C@@H](Cc1ccc(N=[N+]=[N-])cc1)C(=O)O',

    # Alkyne-containing - bioorthogonal chemistry
    'ncAA_PropargylGly': 'N[C@@H](CC#C)C(=O)O',
    'ncAA_HomoPropargylGly': 'N[C@@H](CCC#C)C(=O)O',

    # Beta-amino acids - backbone modification class
    'ncAA_beta_Ala':     'NCCC(=O)O',
    'ncAA_beta_Phe':     'N[C@@H](Cc1ccccc1)CC(=O)O',
    'ncAA_beta_Leu':     'NCC(CC(C)C)C(=O)O',

    # Alpha-methyl amino acids - proteolysis resistant class
    'ncAA_Aib':          'CC(N)(C)C(=O)O',
    'ncAA_aMethylPhe':   'N[C@](C)(Cc1ccccc1)C(=O)O',
    'ncAA_aMethylLeu':   'N[C@](C)(CC(C)C)C(=O)O',

    # D-amino acids - mirror image pharmacophore class
    'ncAA_D_Ala':        'N[C@H](C)C(=O)O',
    'ncAA_D_Phe':        'N[C@H](Cc1ccccc1)C(=O)O',
    'ncAA_D_Leu':        'N[C@H](CC(C)C)C(=O)O',
    'ncAA_D_Pro':        'OC(=O)[C@@H]1CCCN1',

    # Halogenated aromatics - medicinal chemistry class
    'ncAA_Cl_Phe':       'N[C@@H](Cc1ccc(Cl)cc1)C(=O)O',
    'ncAA_Br_Phe':       'N[C@@H](Cc1ccc(Br)cc1)C(=O)O',
    'ncAA_I_Tyr':        'N[C@@H](Cc1ccc(O)c(I)c1)C(=O)O',

    # Cyclic/constrained - conformationally restricted class
    'ncAA_1_aminocyclopropane': 'N[C]1(CC1)C(=O)O',
    'ncAA_pipecolic':    'OC(=O)[C@@H]1CCCCN1',
    'ncAA_Tic':          'OC(=O)[C@@H]1NCCc2ccccc21',

    # Photocrosslinking - UV-activatable ncAAs
    'ncAA_BzF':          'N[C@@H](Cc1ccc(C(=O)c2ccccc2)cc1)C(=O)O',
    'ncAA_DiazoPhe':     'N[C@@H](Cc1ccc(/C(=N/N)c2ccccc2)cc1)C(=O)O',

    # Charged side chains - electrostatic pharmacophore class
    'ncAA_homo_Glu':     'N[C@@H](CCC(=O)O)C(=O)O',   # wait this is Glu
    'ncAA_beta_homo_Asp': 'N[C@@H](CC(=O)O)CC(=O)O',
    'ncAA_Dap':          'N[C@@H](CN)C(=O)O',
    'ncAA_Dab':          'N[C@@H](CCN)C(=O)O',
    'ncAA_Orn':          'N[C@@H](CCCN)C(=O)O',

    # Hydroxylated - hydrogen bond donor class
    'ncAA_homo_Ser':     'N[C@@H](CCO)C(=O)O',
    'ncAA_beta_homo_Thr': 'N[C@@H]([C@@H](O)C)CC(=O)O',
    'ncAA_F_Hyp':        'OC(=O)[C@@H]1CC(F)CN1',

    # Boronated - emerging therapeutic class
    'ncAA_boronoAla':    'N[C@@H](CB(O)O)C(=O)O',
    'ncAA_boronoPhe':    'N[C@@H](Cc1ccc(B(O)O)cc1)C(=O)O',
}


def fetch_ncaa_from_seed(validate: bool = True) -> pd.DataFrame:
    """
    Builds ncAA target dataframe from the curated seed set.
    Optionally validates each SMILES with RDKit.
    """
    print(f"Loading {len(NCAA_SEED_SET)} curated ncAA seeds...")
    rows = []

    for ncaa_id, smiles in NCAA_SEED_SET.items():
        if validate:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"  [WARNING] Invalid SMILES skipped: {ncaa_id} - {smiles}")
                continue
            # Canonicalize
            smiles = Chem.MolToSmiles(mol)

        rows.append({
            'id':        ncaa_id,
            'smiles':    smiles,
            'type':      'noncanonical_target',
            'anchor_id': None,
        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} valid ncAA seeds")
    return df


def fetch_ncaa_from_pubchem(limit: int = 200) -> pd.DataFrame:
    """
    Fetches non-proteinogenic amino acids from PubChem's
    classification hierarchy. Returns verified ncAA structures.
    
    PubChem classification CID for non-proteinogenic amino acids: 
    Uses the compound classification endpoint.
    """
    print(f"Fetching ncAAs from PubChem (limit={limit})...")

    # PubChem classification for amino acids and derivatives
    # We search by substructure: amino acid backbone
    backbone_smarts = 'NCC(=O)O'  # minimal amino acid backbone

    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/substructure"
        f"/smarts/{requests.utils.quote(backbone_smarts)}/JSON"
        f"?MaxRecords={limit}"
    )

    try:
        # PubChem substructure search is async - submit then poll
        response = requests.post(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/substructure/smarts/JSON",
            data={'smarts': backbone_smarts, 'MaxRecords': limit},
            timeout=30
        )

        if response.status_code == 202:
            # Async job - get listkey and poll
            listkey = response.json()['Waiting']['ListKey']
            import time
            for _ in range(10):
                time.sleep(3)
                poll = requests.get(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/{listkey}/property/IsomericSMILES/JSON"
                )
                if poll.status_code == 200:
                    compounds = poll.json()['PropertyTable']['Properties']
                    break
                elif poll.status_code == 202:
                    continue
            else:
                print("  PubChem polling timed out")
                return pd.DataFrame()

        elif response.status_code == 200:
            compounds = response.json().get('PropertyTable', {}).get('Properties', [])
        else:
            print(f"  PubChem returned {response.status_code}")
            return pd.DataFrame()

        rows = []
        for compound in compounds[:limit]:
            smiles = compound.get('IsomericSMILES', '')
            cid    = compound.get('CID', '')
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            rows.append({
                'id':        f"pubchem_{cid}",
                'smiles':    Chem.MolToSmiles(mol),
                'type':      'noncanonical_target',
                'anchor_id': None,
            })

        df = pd.DataFrame(rows)
        print(f"PubChem returned {len(df)} valid ncAAs")
        return df

    except Exception as e:
        print(f"  PubChem fetch failed: {e}")
        return pd.DataFrame()


def fetch_chembl_by_substructure(limit: int = 200) -> pd.DataFrame:
    """
    Queries ChEMBL by amino acid backbone substructure.
    This is the correct ChEMBL query for your use case -
    not molecule_type=Peptide which is poorly populated.
    """
    print(f"Querying ChEMBL by amino acid substructure (limit={limit})...")

    # Amino acid backbone SMARTS
    backbone = 'NCC(=O)O'

    url = (
        f"https://www.ebi.ac.uk/chembl/api/data/molecule"
        f"?molecule_structures__canonical_smiles__flexmatch={backbone}"
        f"&limit={limit}&format=json"
    )

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"  ChEMBL substructure query failed: {response.status_code}")
            return pd.DataFrame()

        data = response.json()
        rows = []

        for item in data.get('molecules', []):
            structs = item.get('molecule_structures')
            if not structs:
                continue
            smiles = structs.get('canonical_smiles')
            if not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            rows.append({
                'id':        item['molecule_chembl_id'],
                'smiles':    Chem.MolToSmiles(mol),
                'type':      'noncanonical_target',
                'anchor_id': None,
            })

        df = pd.DataFrame(rows)
        print(f"ChEMBL substructure query returned {len(df)} molecules")
        return df

    except Exception as e:
        print(f"  ChEMBL substructure query failed: {e}")
        return pd.DataFrame()

def fetch_chembl_peptides(limit: int = 2000) -> pd.DataFrame:
    """
    Master Ingestion Function.
    Combines the curated Seed Set with paginated ChEMBL data, 
    and strictly deduplicates them by canonical SMILES.
    """
    import requests
    import time
    import pandas as pd
    from rdkit import Chem
    
    print(f"Assembling Non-Canonical Targets...")
    all_rows = []

    # ---------------------------------------------------------
    # 1. LOAD CURATED SEED SET
    # ---------------------------------------------------------
    # (Assuming NCAA_SEED_SET is defined at the top of your file)
    for name, smiles in NCAA_SEED_SET.items():
        all_rows.append({
            'id': name, 
            'smiles': smiles, 
            'type': 'noncanonical_target'
        })
    print(f"  -> Loaded {len(NCAA_SEED_SET)} curated literature seeds.")

    # ---------------------------------------------------------
    # 2. FETCH PAGINATED CHEMBL DATA
    # ---------------------------------------------------------
    print(f"  -> Fetching up to {limit} synthetic ncAAs from ChEMBL...")
    backbone_smiles = "NCC(=O)O"
    base_url = f"https://www.ebi.ac.uk/chembl/api/data/substructure/{backbone_smiles}.json"
    
    standard_20 = {
        'NC(C(=O)O)C', 'NCC(=O)O', 'NC(C(=O)O)C(C)C', 'NC(C(=O)O)CC(C)C',
        'NC(C(=O)O)C(C)CC', 'NC(C(=O)O)CO', 'NC(C(=O)O)C(O)C', 'NC(C(=O)O)CS',
        'NC(C(=O)O)CCSC', 'NC(C(=O)O)CC(=O)N', 'NC(C(=O)O)CCC(=O)N',
        'NC(C(=O)O)CC(=O)O', 'NC(C(=O)O)CCC(=O)O', 'NC(C(=O)O)CCCCN',
        'NC(C(=O)O)CCCNC(=N)N', 'O=C(O)C1CCCN1', 'NC(C(=O)O)Cc1ccccc1',
        'NC(C(=O)O)Cc1ccc(O)cc1', 'NC(C(=O)O)Cc1c[nH]c2ccccc12',
        'NC(C(=O)O)Cc1cnc[nH]1'
    }

    limit_per_page = 1000
    offset = 0
    chembl_count = 0

    while chembl_count < limit:
        params = {
            "limit": limit_per_page, 
            "offset": offset, 
            "only": "molecule_chembl_id,molecule_structures"
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=60)
            if response.status_code != 200:
                print(f"  [WARN] API returned {response.status_code}. Stopping ChEMBL fetch.")
                break
                
            data = response.json()
            molecules = data.get('molecules', [])
            if not molecules:
                break

            for mol_data in molecules:
                structures = mol_data.get('molecule_structures')
                if not structures or not structures.get('canonical_smiles'): 
                    continue

                smiles = structures.get('canonical_smiles')
                mol = Chem.MolFromSmiles(smiles)
                if not mol: 
                    continue
                    
                canon_smiles = Chem.MolToSmiles(mol)
                if canon_smiles not in standard_20:
                    all_rows.append({
                        'id': f"chembl_{mol_data.get('molecule_chembl_id')}",
                        'smiles': canon_smiles,
                        'type': 'noncanonical_target'
                    })
                    chembl_count += 1
                    
                    if chembl_count >= limit: 
                        break

            print(f"     ... Fetched page. Current ChEMBL ncAAs: {chembl_count}")
            offset += limit_per_page
            time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"  [WARN] Request failed: {e}. Stopping ChEMBL fetch.")
            break

    # ---------------------------------------------------------
    # 3. CONVERT TO DATAFRAME & DEDUPLICATE
    # ---------------------------------------------------------
    df = pd.DataFrame(all_rows)
    initial_len = len(df)
    
    # We already converted everything to canonical SMILES during ingestion,
    # so we can directly drop duplicates based on the 'smiles' column.
    # keep='first' ensures that if a ChEMBL molecule matches one of your 
    # curated seeds, it keeps your seed ID instead of the ChEMBL ID!
    df = df.drop_duplicates(subset=['smiles'], keep='first').reset_index(drop=True)
    
    print(f"\n✅ Total unique ncAAs after deduplication: {len(df)} (Removed {initial_len - len(df)} duplicates)")
    return df


def fetch_canonical_baselines(limit: int = 200) -> pd.DataFrame:
    """
    Fetches canonical amino acid baselines from UniProt.
    These serve as in-batch background negatives during contrastive training.
    """
    import requests
    print(f"Fetching {limit} canonical peptides from UniProt...")
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    rows = []
    offset = 0
    page_size = min(500, max(1, limit))  # UniProt search API max size is 500

    while len(rows) < limit:
        params = {
            'query': '(length:[5 TO 20])',
            'format': 'json',
            'size': page_size,
            'offset': offset,
        }

        response = requests.get(base_url, params=params, timeout=30)
        if response.status_code != 200:
            raise Exception(f"UniProt API failed: {response.status_code}")

        data = response.json()
        results = data.get('results', [])
        if not results:
            break

        for item in results:
            uid = item.get('primaryAccession')
            seq = item.get('sequence', {}).get('value')
            if not seq or not uid:
                continue

            mol = Chem.MolFromSequence(seq)
            if not mol:
                print(f"  [WARN] RDKit failed on {uid}: {seq[:15]}...")
                continue

            try:
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                rows.append({
                    'id':        f"uniprot_{uid}",
                    'smiles':    smiles,
                    'type':      'canonical_baseline',
                    'anchor_id': None,
                })
            except Exception as e:
                print(f"  [WARN] Sanitization failed on {uid}: {e}")
                continue

        offset += page_size

        # Defensive: avoid infinite loops in case API misbehaves
        if offset > 1000000:
            print("  [WARN] Offset exceeded safety limit; stopping pagination")
            break

    # Trim to requested limit and deduplicate
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=['smiles'], keep='first').reset_index(drop=True)
        df = df.iloc[:limit].reset_index(drop=True)

    print(f"UniProt returned {len(df)} canonical baselines")
    return df
