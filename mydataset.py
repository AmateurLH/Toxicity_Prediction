import os

import numpy as np

from utils import set_random_seed
import pandas as pd

set_random_seed()

# document path
root = os.getcwd()
data_dir = os.path.join(root, 'data')
[os.mkdir(data_dir + "/" + i) for i in ['raw', 'processed'] if not os.path.exists(data_dir + "/" + i)]
raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')


def raw_label_processed( dir=data_dir + '/Toxicity_SMILES.xlsx' ):
	# 1. read raw data
	# 2. process raw data
	# 3. save processed data
	for i in ['rat', 'mouse']:
		for j in ['oral', 'subcutaneous']:
			raw_df = pd.read_excel(dir, sheet_name=f'{i}_{j}', engine='openpyxl',
			                       header=None, names=['SMILES', 'Dose', 'pDose', 'Class'], skiprows=1)
			raw_df['pDose'] = np.log10(raw_df['Dose']).round(3)
			if j == 'oral':  # oral dose
				raw_df['Class'] = raw_df['Dose'].apply(lambda x: 1 if x <= 5
				else (2 if 5 < x <= 50
				      else (3 if 50 < x <= 300
				            else (4 if 300 < x <= 2000
				                  else 5))))
			else:  # subcutaneous dose
				raw_df['Class'] = raw_df['Dose'].apply(lambda x: 1 if x <= 50
				else (2 if 50 < x <= 200
				      else (3 if 200 < x <= 1000
				            else (4 if 1000 < x <= 2000
				                  else 5))))
			# counts = raw_df['class'].value_counts(sort=True)
			# print(counts)
			with open(data_dir + '/' + i + '_' + j + '.csv', 'w', newline='') as f:
				f.write(raw_df.to_csv(index=False, header=True))
			print(f'{i}_{j} is done!')


raw_label_processed()


def add_molecular_name():
	pass


def normalize_pDose():
	pass
