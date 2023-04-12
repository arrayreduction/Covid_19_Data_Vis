import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def import_excel_sheets(sheet_no, skiprows, files, nrows=None):
    '''Import data from multiple like-formated excel sheets.
    sheet_no: 0 indexed sheet number to lookup from excel workbooks
    skiprows: either an integer or tuple indicating which header rows should be skipped 
    nrows: either an integer or a tuple giving the number of rows to read in each sheet
    '''
    dfs = []

    #In the simple case, just interate through the files and append them to an array
    #Where there are tuples, we unpack accordingly

    if (nrows is None and type(skiprows) is int) or (type(nrows) is int and type(skiprows is int)):
        for f in files:
            df = pd.read_excel(f, sheet_name=sheet_no, skiprows=skiprows, nrows=nrows)
            dfs.append(df)

    else:
        for i, f in enumerate(files):
            if type(nrows) is tuple:
                nrow = nrows[i]
            else:
                nrow = nrows
            
            if type(skiprows) is tuple:
                skiprow = skiprows[i]
            else:
                skiprow = skiprows

            df = pd.read_excel(f, sheet_name=sheet_no, skiprows=skiprow, nrows=nrow)
            dfs.append(df)

    return dfs

def get_files(dir, src):
    '''Get list of files with path from given directory'''

    #Build file path according to dir passed
    sub_src = os.path.join(src, dir)

    #Get all filenames with filepath
    files = [os.path.join(dirpath, f) for (dirpath, dirnames, filenames) in os.walk(sub_src) for f in filenames]

    #For saftey, strip any temp files in windows environments
    files = [x for x in files if "~$" not in x]

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

    #Cumulative vaccination
    files_vac = get_files("Cumulative", vaccine_dir)
    files_vac = [x for x in files_vac if x.endswith('.xlsx')]

    #Handle vaccines
    dfs = []
    for i in range(3, 9):
        df = pd.concat(import_excel_sheets(i, 4, files_vac))
        df['sheet_origin'] = i
        dfs.append(df)
    
    df_vacs = pd.concat(dfs, axis=0, sort=False)

    df_vacs.info()

    #Handle covid deaths by age by gender
    
    #All by age
    df_cv19_age_all = pd.concat(import_excel_sheets(7, 6, files_deaths_post22, nrows=(64, 104)))

    #Male by age
    df_cv19_age_male = pd.concat(import_excel_sheets(7, (73, 113), files_deaths_post22, nrows=(64, 104)))

    #Female by age
    df_cv19_age_female = pd.concat(import_excel_sheets(7, (140, 220), files_deaths_post22, nrows=(64, 104)))

    df_cv19_age_all.info()
    df_cv19_age_male.info()
    df_cv19_age_female.info()


if __name__ == '__main__':
    main()