import re
import csv

# attacks_ =['ISSBA', 'WaNet','BATT', 'Blended']
attacks_ =['SIG']
for attack in attacks_:
    exp_dir = '../experiments/exp6_FI_B/Badnet_abl/'  
    # csv_file = '/ABL_BATT'
    files_ = ['1_random_BvB', '2_BvB']
    for csv_file in files_:
        txt_file = exp_dir+csv_file+'.txt'


        import re

        # Input and output file paths
        input_file = exp_dir+csv_file+'.txt'  # replace with your input file path
        output_file = exp_dir+csv_file+'_out.txt'  # replace with your desired output file path

        # Regular expression pattern to match the line and capture ASR and ACC values
        pattern = r'model - ASR:\s*([\d.]+), ACC:\s*([\d.]+)'

        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                match = re.search(pattern, line)
                if match:
                    asr_value = match.group(1)
                    acc_value = match.group(2)
                    # Write the values to the output file
                    outfile.write(f"{asr_value} {acc_value}\n")

    print("Extraction complete! Check output.txt for results.")
