

################# TRAINING FILES #################
# import os
# import csv

# post_fix = 'pt'

# training_names = os.listdir('/media/hdd/CAMELYON16/feats_resnet50/training/' + post_fix + '_files')

# rows = []
# # rows = [['case_id', 'slide_id', 'label']]

# count = 0
# for name in training_names:
# 	type_, post = name.split('_')
# 	num, _ = post.split('.')

# 	num_int = int(num) - 1 if type_ == 'normal' else int(num)+159
# 	# print(type_, 'slide_' + str(num_int) + '.' + post_fix)

# 	rows.append([num_int, 'slide_' + str(num_int), type_])
# 	count += 1
# 	os.rename('/media/hdd/CAMELYON16/feats_resnet50/training/' + post_fix + '_files/' + name,
# 	 '/media/hdd/CAMELYON16/feats_resnet50/training/' + post_fix + '_files/slide_' + str(num_int) + '.' + post_fix)

# def takeSecond(elem):
    # return elem[3]

# rows.sort(key=lambda x: x[0])

# for row in rows:
# 	row[0] = 'patient_' + str(row[0])

# rows.insert(0, ['case_id', 'slide_id', 'label'])

# with open('training_cases.csv', 'w') as file:
# 	writer = csv.writer(file)
# 	writer.writerows(rows)


################# TESTING FILES #################
import os
import csv

post_fix = 'h5'

rows = []
with open('/media/hdd/CAMELYON16/testing/reference.csv', 'r') as file:
	reader = csv.reader(file)

	for row in reader:
		rows.append(row)

new_rows = []

for row in rows:
	num_int = int(row[0].split('_')[1]) + 269
	new_rows.append([num_int, 'slide_'+str(num_int), 'normal' if row[1] == 'Normal' else 'tumor'])
	os.rename('/media/hdd/CAMELYON16/feats_resnet50/testing/' + post_fix + '_files/test_' + row[0].split('_')[1] + '.' + post_fix,
	 '/media/hdd/CAMELYON16/feats_resnet50/testing/' + post_fix + '_files/slide_' + str(num_int) + '.' + post_fix)

# new_rows.sort(key=lambda x: x[0])

# for new_row in new_rows:
# 	new_row[0] = 'patient_' + str(new_row[0])

# print(new_rows)

# with open('testing_cases.csv', 'w') as file:
# 	writer = csv.writer(file)
# 	writer.writerows(new_rows)
