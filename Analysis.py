import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def import_excel_sheets(sheet_no, files):
    dfs = []
    for f in files:
        df = pd.read_excel(f)
        dfs.append(df)

    return dfs

def get_files(dir, src):
    '''Get list of files with path from given directory'''

    #Build file path according to dir passed
    sub_src = os.path.join(src, dir)

    #Get all filenames with filepath
    files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(sub_src) for f in filenames]

    return files

def main():
    deaths_dir = r"C:\Users\yblad\Documents\For Bsc\Year 3\Data Vis\Assessment\Data\Deaths registered by year"
    vaccine_dir = r"C:\Users\yblad\Documents\For Bsc\Year 3\Data Vis\Assessment\Data\Vaccination"

    #Post 2022 deaths
    files_deaths_post22 = get_files("Post22", deaths_dir)
    files_deaths_post22 = [x for x in files_deaths_post22 if x.endswith('.xlsx')]

    #Pre 2022 deaths
    files_deaths_pre22 = get_files("Pre22", deaths_dir)
    files_deaths_pre22 = [x for x in files_deaths_pre22 if x.endswith('.xlsx')]

    #Post2018 vaccination
    files_vac_post18 = get_files("Post2018", vaccine_dir)
    files_vac_post18 = [x for x in files_vac_post18 if x.endswith('.xlsx')]

    #Pre2018 vaccination
    files_vac_pre18 = get_files("Pre2018", vaccine_dir)
    files_vac_pre18 = [x for x in files_vac_pre18 if x.endswith('.xlsx')]

if __name__ == '__main__':
    main()