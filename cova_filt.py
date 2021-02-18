import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesWriter, SDWriter
from functools import reduce

if __name__ == '__main__':
	

	sdf_f='./PostEra_data_25_01.sdf'
	
	smarts=['C=!@CC=[O,S]','[$([CH]),$(CC)]#CC(=O)[C,c]','[$([CH]),$(CC)]#CS(=O)(=O)[C,c]','C=C(C=O)C=O','[$([CH]),$(CC)]#CC(=O)O[C,c]',\
	       '[CX3](=[OX1])C','[OX1]=CN','[C;H2:1]=[C;H1]C(N)=O','Cl[C;H2:1]C(N)=O','[NX3][CX3](=[OX1])[OX2H0]',\
           '[#6]-[#6](=[#8])-[#6](-[#7])=[#8]','[#6][CX3](=O)[OX2H0][#6]','N#[C:1]-[*]',\
           '[#6]=[#7]-[#7]-[#6](=[#8])-[#7](-[#6])-[#6]','N#[C:1]-[*]']
	


	patt=[]
	
	for x in smarts:
		print(x)
		print(type(x))
		patt+=[(x,Chem.MolFromSmarts(x))]
	
	suppl=Chem.SDMolSupplier(sdf_f) # read file
	
	matches={x:[] for x in smarts}

	for mol in suppl:
		for name,pat in patt:
			if mol.HasSubstructMatch(pat):
				matches[name]+=[mol]
	
	#use SmilesWriter if you want the smi format

	mol_w = {x:[] for x in set(reduce(lambda x,y:x+y,list(matches.values())))}
	
	for k in matches.keys():
		print(matches[k])
		for mol in matches[k]:
			mol_w[mol]+=[k]

	writer=SDWriter('PostEra_cov_cmpds_cy.sdf')

	for mol in mol_w.keys():
		#print(help(mol))
		#print(dir(mol))
		#print(mol.GetPropNames())
		print(mol.GetProp('_CID'))
		#print(mol.CID)
		mol.SetProp('Patt','{}'.format(mol_w[mol]))
		if len(mol_w[mol])>1:
			print('molecule {}, len prop = {}'.format(mol.GetProp('_CID'),len(mol_w[mol])))
		writer.write(mol)
	writer.close()
	
	writer=SDWriter('PostEra_Non_cov_cy.sdf')
	for mol in suppl:
		if mol.GetProp('_CID') not in [x.GetProp('_CID') for x in list(mol_w.keys())]:
			writer.write(mol)

	writer.close()
	
	#print(matches)
	
	

