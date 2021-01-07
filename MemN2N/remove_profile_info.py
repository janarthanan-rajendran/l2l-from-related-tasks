in_data_dir = './../data/personalized-dialog-dataset/split-by-profile-10-mp-mixed-s/'
out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-10-mp-mixed-s-wop/'

profiles = ['female_elderly', 'female_middle-aged','female_young', 'male_elderly','male_middle-aged', 'male_young']
file_names = ['personalized-dialog-task5-full-dialogs-trn.txt', 'personalized-dialog-task5-full-dialogs-dev.txt',
              'personalized-dialog-task5-full-dialogs-tst.txt']

for profile in profiles:
    for file_name in file_names:
        with open(in_data_dir+ profile + '/' + file_name,'r') as f_in:
            with open(out_data_dir + profile + '/' + file_name,'w') as f_out:
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

# # checking
# dialog_count = 0
# with open(out_data_dir+file_name,'r') as f_out:
#     out_lines = f_out.readlines()
#     for line in out_lines:
#         line = line.strip()
#         if not line:
#             dialog_count +=1
#
# print(dialog_count)
