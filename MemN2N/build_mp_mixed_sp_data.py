import numpy as np

in_data_dir = './../data/personalized-dialog-dataset/full/'
out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-10-mp-mixed-s/'
file_name = 'personalized-dialog-task5-full-dialogs-trn.txt'
skip_dialog = True
profiles = ['female_elderly', 'female_middle-aged','female_young', 'male_elderly','male_middle-aged', 'male_young']

for profile in profiles:
    print(profile)
    with open(in_data_dir + file_name,'r') as f_in:
        with open(out_data_dir + profile + '/' + file_name,'a') as f_out:
            in_lines = f_in.readlines()
            u_dialogs = []  # list of dialogs with only user utterances
            dialogs = []  # list of full dialogs
            u_dialog = ''
            dialog = ''
            for line_orig in in_lines:
                line = line_orig.strip()
                if line:
                    if '\t' in line:
                        u, r = line.split('\t')
                        u_dialog += u + '\t'
                    else:
                        nid, _ = line.split(' ', 1)
                        if nid == '1':
                            u_dialog += ''.join(line.split(' ')[3:]) + '\t'
                        else:
                            u_dialog += line + '\t'

                    dialog += line_orig
                else:
                    u_dialogs.append(u_dialog)
                    u_dialog = ''
                    dialogs.append(dialog)
                    dialog = ''

            u_dialog_equivalents = []
            u_dialog_equivalent = []
            for u_dialog in u_dialogs:
                for num, u_dialog_temp in enumerate(u_dialogs):
                    if u_dialog_temp == u_dialog:
                        u_dialog_equivalent.append(num)
                u_dialog_equivalents.append(u_dialog_equivalent)  # index of equivalent dialogs
                u_dialog_equivalent = []

            num = 0
            for u_dialog, dialog in zip(u_dialogs, dialogs):
                if not (skip_dialog and len(u_dialog_equivalents[num]) == 1):
                    lines = dialog.split('\n')
                    skip_line = True
                    for line_num, line in enumerate(lines):
                        rand_index = np.random.randint(len(u_dialog_equivalents[num]))
                        selected_line = dialogs[u_dialog_equivalents[num][rand_index]].split('\n')[line_num]
                        if not skip_line:
                            if selected_line.strip():
                                nid, selected_line = selected_line.split(' ',1)
                                nid = int(nid)
                                f_out.write(str(nid-1) + ' ' + selected_line)
                            else:
                                f_out.write(selected_line)
                            f_out.write('\n')
                        skip_line = False
                num += 1

# checking
for profile in profiles:
    dialog_count = 0
    with open(out_data_dir + profile + '/' + file_name,'r') as f_out:
        out_lines = f_out.readlines()
        for line in out_lines:
            line = line.strip()
            if not line:
                dialog_count +=1
    print(profile + ': ', dialog_count)

