import os
import csv

normal_names = os.listdir('/data1/camelyon_data/CAMELYON16/training_patch/normal/patches')
tumor_names = os.listdir('/data1/camelyon_data/CAMELYON16/training_patch/tumor/patches')
te_names = os.listdir('/data1/camelyon_data/CAMELYON16/testing_patch/images/patches')

te_info = []
with open('/data1/camelyon_data/CAMELYON16/testing/reference.csv') as f:
	reader = csv.reader(f)

	for row in reader:
		te_info.append(row)

with open('./transmil_val_id.txt') as f:
	val_names = f.readlines()

val_names = [val_name[:-1] for val_name in val_names]

all_cases = [['case_id', 'slide_id', 'label']]
splits_0_bool = [['', 'train', 'val', 'test']]
splits_0_descriptor = [['', 'train', 'val', 'test']]
splits_0 = [['', 'train', 'val', 'test']]

for i in range(216):
	splits_0.append([i, '', '', ''])

patient_counter = 0
normal_counter = ['normal', 0, 0, 0]
tumor_counter = ['tumor', 0, 0, 0]


independent_index = [1, 1, 1]

for name in normal_names:
	all_cases.append(['patient_'+str(patient_counter), name[:-3], 'normal'])
	if name[:-3] not in val_names:
		splits_0_bool.append([name[:-3], 'TRUE', 'FALSE', 'FALSE'])
		normal_counter[1] += 1
		splits_0[independent_index[0]][1] = name[:-3]
		independent_index[0] += 1
	else:
		splits_0_bool.append([name[:-3], 'FALSE', 'TRUE', 'FALSE'])
		normal_counter[2] += 1
		splits_0[independent_index[1]][2] = name[:-3]
		independent_index[1] += 1

	patient_counter += 1


for name in tumor_names:
	all_cases.append(['patient_'+str(patient_counter), name[:-3], 'tumor'])
	if name[:-3] not in val_names:
		splits_0_bool.append([name[:-3], 'TRUE', 'FALSE', 'FALSE'])
		tumor_counter[1] += 1
		splits_0[independent_index[0]][1] = name[:-3]
		independent_index[0] += 1
	else:
		splits_0_bool.append([name[:-3], 'FALSE', 'TRUE', 'FALSE'])
		tumor_counter[2] += 1
		splits_0[independent_index[1]][2] = name[:-3]
		independent_index[1] += 1

	patient_counter += 1

def get_label_by_test_id(id):
	for elem in te_info:
		if elem[0] == id:
			return elem[1]

for name in te_names:
	label = get_label_by_test_id(name[:-3])

	if label == 'Normal':
		normal_counter[3] += 1
		label = 'normal'
	elif label == 'Tumor':
		tumor_counter[3] += 1
		label = 'tumor'

	all_cases.append(['patient_'+str(patient_counter), name[:-3], label])

	splits_0[independent_index[2]][3] = name[:-3]
	independent_index[2] += 1

	splits_0_bool.append([name[:-3], 'FALSE', 'FALSE', 'TRUE'])

	patient_counter += 1


splits_0_descriptor.append(normal_counter)
splits_0_descriptor.append(tumor_counter)


# print(all_cases)
# print(splits_0_bool)
# print(splits_0_descriptor)
# print(patient_counter)
# print(splits_0)

to_write = [
[all_cases, 'all_cases.csv'],
[splits_0_bool, 'splits_0_bool.csv'],
[splits_0_descriptor, 'splits_0_descriptor.csv'],
[splits_0, 'splits_0.csv']]

for i in range(len(to_write)):
	print('***', i)
	print(to_write[i][0])
	with open(to_write[i][1], 'w') as f:
		writer = csv.writer(f)
		writer.writerows(to_write[i][0])
