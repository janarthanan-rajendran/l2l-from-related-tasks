in_data_dir = './../data/personalized-dialog-dataset/split-by-profile/male_young/'
out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-10/male_young/'
file_name = 'personalized-dialog-task5-full-dialogs-trn.txt'

train_data_percent = 10
dialog_count = 0
dialog_max_count = train_data_percent/100 * 1000

with open(in_data_dir+file_name,'r') as f_in:
    with open(out_data_dir+file_name,'w') as f_out:
        in_lines = f_in.readlines()
        for line in in_lines:
            f_out.write(line)
            line = line.strip()
            if not line:
                dialog_count +=1
                if dialog_count == dialog_max_count:
                    break

# checking
dialog_count = 0
with open(out_data_dir+file_name,'r') as f_out:
    out_lines = f_out.readlines()
    for line in out_lines:
        line = line.strip()
        if not line:
            dialog_count +=1

print(dialog_count)
