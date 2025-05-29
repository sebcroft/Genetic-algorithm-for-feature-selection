

class ProcessData:
    def __init__(self):
        
        self.moieties = {'P': "C1=CC=C(C=C1)",
                        "m" : "C1=CC=CC(=C1)",
                        "o" : "C1=CC=CC=C1",            
                        'D' : "C1=CC=C(C=C1)C1=CC=C(C=C1)",
                        'N' : "C1=CC=C(C2=C1)C=C(C=C2)"
                        , '(rN)': "c1C=CC(=C(C=CC=C2)c12)"
                        , '(qN)': "C1=C2C(=CC=C1)C(=CC=C2)"
                        , '(Tr)' : "N1C=C(N=N1)"
                        , '(mDp)' : "C1=CC(=CC=C1)C(=Cc2)C=Cc2"
                        , '(Dm)' : "C1=CC(=CC=C1)C(=CC=C2)C=C2"
                        , '(Do)' : "c1ccccc1c2ccccc2",
                        "C" : "C",
                        "E" : "O",
                        "K" : "C(=O)",
                        "S" : "S(=O)(=O)",
                        "d" : "C(=O)C(=O)",
                        "A" : "N"}


    def drop_ps(self, name):
        return 'P' + name.replace('P', '')



    def hyphen_to_brackets(self, frag):
        """
        frag = 'E-Tr-C'
        hyphen_to_brackets(frag)
        > returns: 'E(Tr)C'
        """
        spfrag = frag.split('-')
        return ''.join(f'({i})' if len(i) > 1 else i for i in spfrag)
    

    def _split_name(self, name):
        """
        Function will separate characters like: PEPE(qN)K -> [P , E , P , E , (qN) , K] i.e. splits all characters but keeps characters in brackets together
        
        Inputs
        ------
        name : str
            String that contains groups like 
        
        Returns
        -------
        List
            A list of strings
        
        Example usage
        -------------
        gd = GetData()
        gd._split_name('PE(rN)KmK(qN)(aa)')
        """

        # Step 1: Split the string into individual characters
        chars = list(name)
        
        # Step 2: Initialize variables
        merged = []
        temp = []
        inside_parentheses = False
        
        # Step 3: Loop through each character
        for char in chars:
            if char == '(':
                inside_parentheses = True
                temp.append(char)
            elif char == ')':
                temp.append(char)
                merged.append(''.join(temp))
                temp = []
                inside_parentheses = False
            elif inside_parentheses:
                temp.append(char)
            else:
                merged.append(char)
        return merged
       

    def to_smiles(self, keyname, end_cap='*'):
        """
        Converts the key name into the corresponding SMILES string using self.moieties.
    
        Parameters
        ----------
        keyname : str
            A string containing key moieties (should be present in self.moieties).
        end_cap : str, optional
            SMILES string to be applied to both ends of the molecule (default: '*').
    
        Returns
        -------
        str
            The corresponding SMILES string.
    
        Raises
        ------
        KeyError
            If any keymoiety in `keyname` is not found in `self.moieties`.
        """
        spl_keynames = self._split_name(keyname)
        
        # Find missing moieties
        missing_moieties = [keymoiety for keymoiety in spl_keynames if keymoiety not in self.moieties]
    
        if missing_moieties:
            raise KeyError(f"Missing moieties in self.moieties: {missing_moieties}")
    
        return end_cap + ''.join(self.moieties[keymoiety] for keymoiety in spl_keynames) + end_cap