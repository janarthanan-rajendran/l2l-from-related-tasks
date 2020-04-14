in_data_dir = './../data/personalized-dialog-dataset/full/'
out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-25-mp/female_middle-aged/'
file_name = 'personalized-dialog-task5-full-dialogs-trn.txt'

with open(in_data_dir+file_name,'r') as f_in:
    with open(out_data_dir+file_name,'a') as f_out:
        in_lines = f_in.readlines()
        skip_line = True
        for line in in_lines:
            if not skip_line:
                if line.strip():
                    nid, line = line.split(' ',1)
                    nid = int(nid)
                    f_out.write(str(nid-1) + ' ' + line)
                else:
                    f_out.write(line)
            skip_line = False
            line = line.strip()
            if not line:
                skip_line = True

# checking
dialog_count = 0
with open(out_data_dir+file_name,'r') as f_out:
    out_lines = f_out.readlines()
    for line in out_lines:
        line = line.strip()
        if not line:
            dialog_count +=1

print(dialog_count)