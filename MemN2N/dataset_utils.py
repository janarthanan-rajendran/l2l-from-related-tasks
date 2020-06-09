from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('function_name', 'None', 'create_split_by_profile_from_full')

profiles = ['female_elderly', 'female_middle-aged','female_young', 'male_elderly','male_middle-aged', 'male_young']

def create_split_by_profile_from_full(data_in_dir, data_out_dir):
    trn_filename = 'personalized-dialog-task5-full-dialogs-trn.txt'
    for profile in profiles:
        print(profile)
        with open(data_in_dir+trn_filename, 'r') as f_in:
            with open(data_out_dir+profile+'/'+trn_filename, 'w') as f_out:
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

def build_mp_sp_data(in_data_dir, out_data_dir, file_name, train_data_percent):

    for data_percent in train_data_percent:
        print('data percent', data_percent)
        for profile in profiles:
            print(profile)
            with open(in_data_dir + file_name, 'r') as f_in:
                with open(out_data_dir+'-'+str(data_percent)+ '-mp'+'/'+profile+'/'+file_name, 'a') as f_out:
                    in_lines = f_in.readlines()
                    skip_line = True
                    for line in in_lines:
                        if not skip_line:
                            if line.strip():
                                nid, line = line.split(' ', 1)
                                nid = int(nid)
                                f_out.write(str(nid - 1) + ' ' + line)
                            else:
                                f_out.write(line)
                        skip_line = False
                        line = line.strip()
                        if not line:
                            skip_line = True

            # checking
            dialog_count = 0
            with open(out_data_dir+'-'+str(data_percent)+ '-mp'+'/'+profile+'/'+file_name, 'r') as f_out:
                out_lines = f_out.readlines()
                for line in out_lines:
                    line = line.strip()
                    if not line:
                        dialog_count += 1
            print(dialog_count)


def main(argv):
    if FLAGS.function_name == 'split_by_profile_from_full':
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
        train_data_percent = [5, 10, 25]
        build_mp_sp_data(in_data_dir, out_data_dir, file_name, train_data_percent)
    else:
        print('function not found!')

if __name__ == '__main__':
    app.run(main)