from absl import app
from absl import flags
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string('function_name', 'None', 'create_split_by_profile_from_full')

profiles = ['female_elderly', 'female_middle-aged','female_young', 'male_elderly','male_middle-aged', 'male_young']

def create_split_by_profile_from_full(data_in_dir, data_out_dir):
    filenames = ['personalized-dialog-task5-full-dialogs-trn.txt', 'personalized-dialog-task5-full-dialogs-dev.txt',
                 'personalized-dialog-task5-full-dialogs-tst.txt', 'personalized-dialog-task5-full-dialogs-tst-OOV.txt']
    for profile in profiles:
        print(profile)
        for filename in filenames:
            print(filename)
            with open(data_in_dir+filename, 'r') as f_in:
                with open(data_out_dir+profile+'/'+filename, 'w') as f_out:
                    dialog_count = 0
                    in_lines = f_in.readlines()
                    for line in in_lines:
                        # if line.strip():
                        words = line.split(' ')
                        if len(words) > 0:
                            if words[0] == '1':
                                if words[1] + '_' + words[2] == profile:
                                    write = True
                                    dialog_count +=1
                                else:
                                    write = False
                            if write:
                                f_out.write(line)
            print(dialog_count)

def vary_data_ratio(in_data_dir, out_data_dir, file_name, train_data_percent, total_dialogs):
    for data_percent in train_data_percent:
        dialog_max_count = data_percent/100 * total_dialogs
        for profile in profiles:
            print(profile, data_percent)
            dialog_count = 0
            with open(in_data_dir+profile+'/'+file_name,'r') as f_in:
                with open(out_data_dir+'-'+str(data_percent)+'/'+profile+'/'+file_name,'w') as f_out:
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
            with open(out_data_dir+'-'+str(data_percent)+'/'+profile+'/'+file_name,'r') as f_out:
                out_lines = f_out.readlines()
                for line in out_lines:
                    line = line.strip()
                    if not line:
                        dialog_count +=1
            print(dialog_count)

def build_mp_sp_data(in_data_dir, out_data_dir, file_name, train_data_percent, shift_nid):

    for data_percent in train_data_percent:
        print('data percent', data_percent)
        for profile in profiles:
            print(profile)
            with open(in_data_dir + file_name, 'r') as f_in:
                with open(out_data_dir+'-'+str(data_percent)+ '-mp'+'/'+profile+'/'+file_name, 'a') as f_out:
                    in_lines = f_in.readlines()
                    if shift_nid:
                        skip_line = True
                    else:
                        skip_line = False
                    for line in in_lines:
                        if not skip_line:
                            if line.strip():
                                nid, line = line.split(' ', 1)
                                nid = int(nid)
                                if shift_nid:
                                    f_out.write(str(nid - 1) + ' ' + line)
                                else:
                                    f_out.write(str(nid) + ' ' + line)
                            else:
                                f_out.write(line)
                        skip_line = False
                        line = line.strip()
                        if not line:
                            if shift_nid:
                                skip_line = True
                            else:
                                skip_line = False

            # checking
            dialog_count = 0
            with open(out_data_dir+'-'+str(data_percent)+ '-mp'+'/'+profile+'/'+file_name, 'r') as f_out:
                out_lines = f_out.readlines()
                for line in out_lines:
                    line = line.strip()
                    if not line:
                        dialog_count += 1
            print(dialog_count)


def build_mp_mixed_sp_data(in_data_dir, in_data_dir_sp, out_data_dir, file_name, skip_dialog):
    np.random.seed(0)
    for profile in profiles:
        print(profile)
        with open(in_data_dir + file_name, 'r') as f_in:
            with open(in_data_dir_sp + profile + '/' + file_name, 'r') as f_in_sp:  # split by profile dir
                with open(out_data_dir + profile + '/' + file_name, 'a') as f_out:  # has the profile specific data
                    in_lines = f_in.readlines()  # from the 'full' dir
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
                            else:  # grab food preference and DB options
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

                    in_lines_sp = f_in_sp.readlines()
                    u_dialogs_sp = []  # list of dialogs with only user utterances from sp
                    dialogs_sp = []  # list of full dialogs
                    u_dialog = ''
                    dialog = ''
                    for line_orig in in_lines_sp:
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
                            u_dialogs_sp.append(u_dialog)
                            u_dialog = ''
                            dialogs_sp.append(dialog)
                            dialog = ''

                    ## For each dialog in 'full' what are the equivalent dialogs (matching user and db) in 'full'
                    equi_dialog_counts = 0
                    u_dialog_equivalents = []
                    u_dialog_equivalent = []
                    for u_dialog in u_dialogs:
                        for num, u_dialog_temp in enumerate(u_dialogs):
                            if u_dialog_temp == u_dialog:
                                u_dialog_equivalent.append(num)
                        u_dialog_equivalents.append(u_dialog_equivalent)  # index of equivalent dialogs
                        if len(u_dialog_equivalent) > 1:
                            equi_dialog_counts += 1
                        u_dialog_equivalent = []
                    print('Dialogs in full with multiple equi dialogs: ', equi_dialog_counts)

                    ## For each dialog in sp, what are equivalent dialogs in 'full'
                    u_dialog_equivalents_sp = []
                    u_dialog_equivalent = []
                    for u_dialog in u_dialogs_sp:
                        for num, u_dialog_temp in enumerate(u_dialogs):
                            if u_dialog_temp == u_dialog:
                                u_dialog_equivalent.append(num)
                        u_dialog_equivalents_sp.append(u_dialog_equivalent)  # index of equivalent dialogs
                        u_dialog_equivalent = []

                    ## if skip-dialog=False, for all the dialogs in the 'full' mix when equivalent dialogs available and mix and write
                    ## to SP(out) file
                    num = 0
                    for u_dialog, dialog in zip(u_dialogs, dialogs):  # from 'full'
                        if not (skip_dialog and len(u_dialog_equivalents[num]) == 1):
                            lines = dialog.split('\n')
                            skip_line = True
                            for line_num, line in enumerate(lines):
                                rand_index = np.random.randint(len(u_dialog_equivalents[num]))
                                selected_line = dialogs[u_dialog_equivalents[num][rand_index]].split('\n')[line_num]
                                if not skip_line:
                                    if selected_line.strip():
                                        nid, selected_line = selected_line.split(' ', 1)
                                        nid = int(nid)
                                        f_out.write(str(nid - 1) + ' ' + selected_line)
                                    else:
                                        f_out.write(selected_line)
                                    f_out.write('\n')
                                skip_line = False
                        num += 1
                    """
                    ## for each dialog in sp, find equivalent dialogs in 'full', then mix and write 
                    ## This will lead to more confusion, hence more harmful related tasks
                    for i in range(10):
                        num = 0
                        skipped_dialog_count = 0
                        for u_dialog, dialog in zip(u_dialogs_sp, dialogs_sp):
                            if not (skip_dialog and len(u_dialog_equivalents_sp[num]) <= 1):
                                lines = dialog.split('\n')
                                skip_line = True
                                for line_num, line in enumerate(lines):
                                    rand_index = np.random.randint(len(u_dialog_equivalents_sp[num]))
                                    selected_line = dialogs_sp[u_dialog_equivalents_sp[num][rand_index]].split('\n')[
                                        line_num]
                                    if not skip_line:
                                        if selected_line.strip():
                                            nid, selected_line = selected_line.split(' ', 1)
                                            nid = int(nid)
                                            f_out.write(str(nid - 1) + ' ' + selected_line)
                                        else:
                                            f_out.write(selected_line)
                                        f_out.write('\n')
                                    skip_line = False
                            else:
                                skipped_dialog_count += 1
                            num += 1
                        print('skipped dialog count sp', skipped_dialog_count)
                    """

    # checking
    for profile in profiles:
        dialog_count = 0
        with open(out_data_dir + profile + '/' + file_name, 'r') as f_out:
            out_lines = f_out.readlines()
            for line in out_lines:
                line = line.strip()
                if not line:
                    dialog_count += 1
        print(profile + ': ', dialog_count)


def build_mp_mixed_full_data(in_data_dir, out_data_dir, file_name, skip_dialog):
    np.random.seed(0)
    with open(in_data_dir + file_name, 'r') as f_in:
        with open(out_data_dir + file_name, 'w') as f_out:  # has the profile specific data
            in_lines = f_in.readlines()  # from the 'full' dir
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
                    else:  # grab food preference and DB options
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

            ## For each dialog in 'full' what are the equivalent dialogs (matching user and db) in 'full'
            equi_u_dialog_counts = 0
            u_dialog_equivalents = []
            u_dialog_equivalent = []
            for u_dialog in u_dialogs:
                for num, u_dialog_temp in enumerate(u_dialogs):
                    if u_dialog_temp == u_dialog:
                        u_dialog_equivalent.append(num)
                u_dialog_equivalents.append(u_dialog_equivalent)  # index of equivalent dialogs
                if len(u_dialog_equivalent) > 1:
                    equi_u_dialog_counts += 1
                u_dialog_equivalent = []
            print('Dialogs in full with multiple user equi dialogs: ', equi_u_dialog_counts)

            ## For each dialog in 'full' what are the equivalent dialogs (matching full dialogs) in 'full'
            equi_dialog_counts = 0
            dialog_equivalents = []
            dialog_equivalent = []
            for dialog in dialogs:
                for num, dialog_temp in enumerate(dialogs):
                    if dialog_temp == dialog:
                        dialog_equivalent.append(num)
                dialog_equivalents.append(dialog_equivalent)  # index of equivalent dialogs
                if len(dialog_equivalent) > 1:
                    equi_dialog_counts += 1
                dialog_equivalent = []
            print('Dialogs in full with multiple full equi dialogs: ', equi_dialog_counts)


            ## if skip-dialog=False, for all the dialogs in the 'full' mix when equivalent dialogs available and mix and write
            ## to out file
            num = 0
            for u_dialog, dialog in zip(u_dialogs, dialogs):  # from 'full'
                if not (skip_dialog and len(u_dialog_equivalents[num]) == 1):
                    lines = dialog.split('\n')
                    skip_line = True
                    for line_num, line in enumerate(lines):
                        rand_index = np.random.randint(len(u_dialog_equivalents[num]))
                        selected_line = dialogs[u_dialog_equivalents[num][rand_index]].split('\n')[line_num]
                        if not skip_line:
                            if selected_line.strip():
                                nid, selected_line = selected_line.split(' ', 1)
                                nid = int(nid)
                                f_out.write(str(nid - 1) + ' ' + selected_line)
                            else:
                                f_out.write(selected_line)
                            f_out.write('\n')
                        skip_line = False
                num += 1
            """
            ## for each dialog in sp, find equivalent dialogs in 'full', then mix and write 
            ## This will lead to more confusion, hence more harmful related tasks
            for i in range(10):
                num = 0
                skipped_dialog_count = 0
                for u_dialog, dialog in zip(u_dialogs_sp, dialogs_sp):
                    if not (skip_dialog and len(u_dialog_equivalents_sp[num]) <= 1):
                        lines = dialog.split('\n')
                        skip_line = True
                        for line_num, line in enumerate(lines):
                            rand_index = np.random.randint(len(u_dialog_equivalents_sp[num]))
                            selected_line = dialogs_sp[u_dialog_equivalents_sp[num][rand_index]].split('\n')[
                                line_num]
                            if not skip_line:
                                if selected_line.strip():
                                    nid, selected_line = selected_line.split(' ', 1)
                                    nid = int(nid)
                                    f_out.write(str(nid - 1) + ' ' + selected_line)
                                else:
                                    f_out.write(selected_line)
                                f_out.write('\n')
                            skip_line = False
                    else:
                        skipped_dialog_count += 1
                    num += 1
                print('skipped dialog count sp', skipped_dialog_count)
            """

    # checking
    dialog_count = 0
    with open(out_data_dir + file_name, 'r') as f_out:
        out_lines = f_out.readlines()
        for line in out_lines:
            line = line.strip()
            if not line:
                dialog_count += 1
    print('dialog count: ', dialog_count)


def remove_profile_info(in_data_dir, out_data_dir, file_names, train_data_percent):
    for data_percent in train_data_percent:
        for profile in profiles:
            for file_name in file_names:
                with open(in_data_dir + '-' + str(data_percent) + '-mp' + '/' + profile + '/' + file_name, 'r') as f_in:
                    with open(out_data_dir + '-' + str(data_percent) + '-mp-wop' + '/' + profile + '/' + file_name, 'w') as f_out:
                        in_lines = f_in.readlines()
                        profile_info = False
                        for line in in_lines:
                            if line.strip():
                                nid, line = line.split(' ', 1)
                                nid = int(nid)
                                if nid == 1:
                                    if '\t' not in line:
                                        profile_info = True
                                    else:
                                        profile_info = False
                                if not profile_info:
                                    f_out.write(str(nid) + ' ' + line)
                                else:
                                    if nid != 1:
                                        f_out.write(str(nid - 1) + ' ' + line)
                            else:
                                f_out.write(line)


def main(argv):
    if FLAGS.function_name == 'create_split_by_profile_from_full':
        ## for creating split-by-profile-from-full datasets
        data_in_dir = './../data/personalized-dialog-dataset/full/'
        data_out_dir = './../data/personalized-dialog-dataset/split-by-profile-from-full/'
        create_split_by_profile_from_full(data_in_dir, data_out_dir)
    elif FLAGS.function_name == 'vary_data_ratio':
        ## for creating datasets with different proportion of training data
        in_data_dir = './../data/personalized-dialog-dataset/split-by-profile-from-full/'
        out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-from-full'
        file_name = 'personalized-dialog-task5-full-dialogs-trn.txt'
        train_data_percent = [5, 10, 25]
        total_dialogs = 2000  # 1000 for split-by-profile and 2000 for split-by-profile-from-full
        vary_data_ratio(in_data_dir, out_data_dir, file_name, train_data_percent, total_dialogs)
    elif FLAGS.function_name == 'build_mp_sp_data':
        ## combining multi-profile and specific profile data
        in_data_dir = './../data/personalized-dialog-dataset/full/'
        out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-from-full'
        file_name = 'personalized-dialog-task5-full-dialogs-trn.txt'
        train_data_percent = [5, 10, 25, 100]
        shift_nid = True
        build_mp_sp_data(in_data_dir, out_data_dir, file_name, train_data_percent, shift_nid)
    # elif FLAGS.function_name == 'build_mp_mixed_sp_data':
    #     in_data_dir = './../data/personalized-dialog-dataset/full/'
    #     in_data_dir_sp = './../data/personalized-dialog-dataset/split-by-profile-25/'
    #     out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-25-mp-mixed/'
    #     file_name = 'personalized-dialog-task5-full-dialogs-trn.txt'
    #     skip_dialog = False
    #     build_mp_mixed_sp_data(in_data_dir, in_data_dir_sp, out_data_dir, file_name, skip_dialog)
    elif FLAGS.function_name == 'build_mp_mixed_sp_data':
        in_data_dir = './../data/personalized-dialog-dataset/full/'
        out_data_dir = './../data/personalized-dialog-dataset/full-mixed/'
        file_name = 'personalized-dialog-task5-full-dialogs-trn.txt'
        skip_dialog = False
        build_mp_mixed_full_data(in_data_dir, out_data_dir, file_name, skip_dialog)
        ## combining multi-profile and specific profile data
        in_data_dir = './../data/personalized-dialog-dataset/full-mixed/'
        out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-from-full-mixed'
        file_name = 'personalized-dialog-task5-full-dialogs-trn.txt'
        train_data_percent = [5, 10, 25, 100]
        # train_data_percent = [5]
        shift_nid = False
        build_mp_sp_data(in_data_dir, out_data_dir, file_name, train_data_percent, shift_nid)
    elif FLAGS.function_name == 'remove_profile_info':
        in_data_dir = './../data/personalized-dialog-dataset/split-by-profile-from-full-mixed'
        out_data_dir = './../data/personalized-dialog-dataset/split-by-profile-from-full-mixed'
        file_names = ['personalized-dialog-task5-full-dialogs-trn.txt',
                      'personalized-dialog-task5-full-dialogs-dev.txt',
                      'personalized-dialog-task5-full-dialogs-tst.txt']
        train_data_percent = [5, 10, 25, 100]
        # train_data_percent = [5]
        remove_profile_info(in_data_dir, out_data_dir, file_names, train_data_percent)
    else:
        print('function not found!')

if __name__ == '__main__':
    app.run(main)