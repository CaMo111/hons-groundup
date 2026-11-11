import torch
from torch_geometric.utils import degree
import math
import re 

def affinitydict():
    print("Passing Indexes to Create Scaled Affinity Score 0~ - 15")
    pk = {}
    pk_p = {}

    unit_mults = {
        'mM': 1e-3,
        'uM': 1e-6,
        'nM': 1e-9,
        'pM': 1e-12,
        'fM': 1e-15
    }

    pattern = re.compile(r'^(?:Kd|Ki|IC50)([<>=~]+)([0-9\.]+)(mM|uM|nM|pM|fM)$')
            
    with open('./database/INDEX_general_PL.2020R1.lst', 'r') as f:
        for idx, line in enumerate(f): 
            if idx in range(0,6):
                continue
            
            parts=line.split()
            pdb_code=parts[0]
            bindingstr=parts[3]

            match = pattern.match(bindingstr)
            if match!=None:
                val=float(match.group(2))
                unit=match.group(3)
                molval = val*unit_mults[unit]

                # cast from 0.0000 -> range between 0.~ to 15~ 
                pk[pdb_code] = -math.log10(molval)

        return pk

        # min_pk = min(pk.values())
        # max_pk = max(pk.values())
        # pk_range = max_pk - min_pk

        # #print(min_pk, max_pk)
        # #print('range=', pk_range)
        
        # for key, value in pk.items():
        #     pk_p[key] = (value-min_pk) / pk_range

        # return pk_p

        """
        Distribution of scaled values of pk_p:
        0.0-0.1:  0.49% (94 items)
        0.1-0.2:  4.84% (921 items)
        0.2-0.3: 15.85% (3018 items)
        0.3-0.4: 26.07% (4962 items)
        0.4-0.5: 29.70% (5654 items)
        0.5-0.6: 18.22% (3468 items)
        0.6-0.7:  3.93% (748 items)
        0.7-0.8:  0.78% (148 items)
        0.8-0.9:  0.09% (17 items)
        0.9-1.0:  0.04% (7 items)
        """


