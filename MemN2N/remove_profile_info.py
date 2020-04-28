in_data_dir = './../data/personalized-dialog-dataset/split-by-profile-25-mp-mixed/male_elderly/'
out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-25-mp-mixed-wop/male_elderly/'
file_name = 'personalized-dialog-task5-full-dialogs-trn.txt'

with open(in_data_dir+file_name,'r') as f_in:
    with open(out_data_dir+file_name,'w') as f_out:
        in_lines = f_in.readlines()
        for line in in_lines:
                if line.strip():
                    nid, line = line.split(' ',1)
                    nid = int(nid)
                    if nid == 1 and ('\t' not in line):
                        profile_info = True
                    if not profile_info:
                        f_out.write(str(nid) + ' ' + line)
                    else:
                        if nid != 1:
                            f_out.write(str(nid-1) + ' ' + line)
                else:
                    f_out.write(line)

# checking
dialog_count = 0
with open(out_data_dir+file_name,'r') as f_out:
    out_lines = f_out.readlines()
    for line in out_lines:
        line = line.strip()
        if not line:
            dialog_count +=1

print(dialog_count)