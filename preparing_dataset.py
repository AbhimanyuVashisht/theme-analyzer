
print('REFACTORING THE TRAINING DATA.....')
train = open('amazon_reviews/train/train.ft.txt', 'r')
fp_train_txt = open('amazon_reviews/train_txt.txt', 'w+')
fp_train_label = open('amazon_reviews/train_label.txt', 'w+')
for i, line in enumerate(train):
    filename = str(i+1) + '.txt'
    f_temp = open('amazon_reviews/train_txt/' + filename, 'w+')
    f_temp.writelines(line.split(':')[1].strip())
    fp_train_label.writelines('\n' + line.split(':')[0][9])
    fp_train_txt.writelines('\n' + filename)
    f_temp.close()
fp_train_label.close()
fp_train_txt.close()

print('REFACTORING THE TESTING DATA.....')
test = open('amazon_reviews/test/test.ft.txt', 'r')
fp_test_txt = open('amazon_reviews/test_txt.txt', 'w+')
fp_test_label = open('amazon_reviews/test_label.txt', 'w+')

for i, line in enumerate(test):
    filename = str(i+1) + '.txt'
    f_temp = open('amazon_reviews/test_txt/' + filename, 'w+')
    f_temp.writelines(line.split(':')[1].strip())
    fp_test_label.writelines('\n' + line.split(':')[0][9])
    fp_test_txt.writelines('\n' + filename)
    f_temp.close()
fp_test_label.close()
fp_test_txt.close()