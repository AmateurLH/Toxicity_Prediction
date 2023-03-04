from mydataset import MyDataSets
from torch_geometric.loader import DataLoader
import utils
from mydataset import generate_raw_train_data

root, data_dir, raw_dir, processed_dir = utils.generate_raw_processed_dir()


train_data_dir, test_data_dir = generate_raw_train_data('rat', 'oral')

train_dataset = MyDataSets(root, train_data_dir, test=False)
test_dataset = MyDataSets(root, test_data_dir, test=True)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

print(train_dataset.get(0))