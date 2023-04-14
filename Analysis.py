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

def import_files(verbose=False):
    '''Import vaccinations and deaths.'''
    #Determine whether we are running under windows native or WSL2. If WSL, use the mount point
    if os.name == 'nt':
        deaths_dir = r"C:\Users\yblad\Documents\For Bsc\Year 3\Data Vis\Assessment\Data\Deaths registered by year"
        vaccine_dir = r"C:\Users\yblad\Documents\For Bsc\Year 3\Data Vis\Assessment\Data\Vaccination"
    elif os.name == 'posix':
        deaths_dir = r"/mnt/c/Documents and Settings/yblad/Documents/For Bsc/Year 3/Data Vis/Assessment/Data/Deaths registered by year"
        vaccine_dir = r"/mnt/c/Documents and Settings/yblad/Documents/For Bsc/Year 3/Data Vis/Assessment/Data/Vaccination"


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

    #Handle covid deaths by age by gender
    
    #All by age
    df_cv19_age_all = pd.concat(import_excel_sheets(7, 6, files_deaths_post22, nrows=(64, 104)))
    df_de_age_all = pd.concat(import_excel_sheets(4, 6, files_deaths_post22, nrows=(12, 52)))

    #Male by age
    df_cv19_age_male = pd.concat(import_excel_sheets(7, (73, 113), files_deaths_post22, nrows=(64, 104)))
    df_de_age_male = pd.concat(import_excel_sheets(4, (21, 61), files_deaths_post22, nrows=(12, 52)))

    #Female by age
    df_cv19_age_female = pd.concat(import_excel_sheets(7, (140, 220), files_deaths_post22, nrows=(64, 104)))
    df_de_age_female = pd.concat(import_excel_sheets(4, (36, 116), files_deaths_post22, nrows=(12, 52)))

    #Handle deaths by region
    df_cv19_region = pd.concat(import_excel_sheets(15, 5, files_deaths_post22))
    df_de_region = pd.concat(import_excel_sheets(14, 6, files_deaths_post22, nrows=(12, 52)))

    #The 2021 data was included in the 2020 for age/sex and region. To get 2020 data
    #We must pull in the 2020 file. This needs to be handled seperately as it has a
    #transposed format

    #Covid 19 occurances

    files_deaths_pre22 = [x for x in files_deaths_pre22 if "2020" in x]
    df_2020_cv = pd.read_excel(*files_deaths_pre22, sheet_name=6, skiprows=4, header=1, nrows=80)
    df_2020_cv.dropna(how='all', inplace=True)
    df_2020_cv.drop(columns='Week ended', inplace=True)

    df_2020_cv_all = df_2020_cv[3:23]
    df_2020_cv_all.set_index('Unnamed: 1', inplace=True)
    df_2020_cv_all = df_2020_cv_all.T
    df_2020_cv_all.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_cv_all.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_cv_males = df_2020_cv[25:45]
    df_2020_cv_males.set_index('Unnamed: 1', inplace=True)
    df_2020_cv_males = df_2020_cv_males.T
    df_2020_cv_males.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_cv_males.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_cv_females = df_2020_cv[47:67]
    df_2020_cv_females.set_index('Unnamed: 1', inplace=True)
    df_2020_cv_females = df_2020_cv_females.T
    df_2020_cv_females.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_cv_females.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_cv_regions = df_2020_cv[68:78]
    df_2020_cv_regions.set_index('Unnamed: 1', inplace=True)
    df_2020_cv_regions = df_2020_cv_regions.T
    df_2020_cv_regions.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_cv_regions.rename(columns={'East':'East of England'}, inplace=True)

    #All registered deaths

    df_2020_cv = pd.read_excel(*files_deaths_pre22, sheet_name=4, skiprows=4, header=1, nrows=92)
    df_2020_cv.dropna(how='all', inplace=True)
    df_2020_cv.drop(columns='Week ended', inplace=True)

    df_2020_de_all = df_2020_cv[13:33]
    df_2020_de_all.set_index('Unnamed: 1', inplace=True)
    df_2020_de_all = df_2020_de_all.T
    df_2020_de_all.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_de_all.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_de_males = df_2020_cv[35:55]
    df_2020_de_males.set_index('Unnamed: 1', inplace=True)
    df_2020_de_males = df_2020_de_males.T
    df_2020_de_males.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_de_males.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_de_females = df_2020_cv[57:77]
    df_2020_de_females.set_index('Unnamed: 1', inplace=True)
    df_2020_de_females = df_2020_de_females.T
    df_2020_de_females.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_de_females.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_de_regions = df_2020_cv[78:89]
    df_2020_de_regions.set_index('Unnamed: 1', inplace=True)
    df_2020_de_regions = df_2020_de_regions.T
    df_2020_de_regions.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_de_regions.rename(columns={'East':'East of England'}, inplace=True)

    #Merge 2020 frames with newer format frames

    df_cv19_age_all = pd.concat([df_cv19_age_all, df_2020_cv_all])
    df_cv19_age_male = pd.concat([df_cv19_age_male, df_2020_cv_males])
    df_cv19_age_female = pd.concat([df_cv19_age_female, df_2020_cv_females])
    df_cv19_region = pd.concat([df_cv19_region, df_2020_cv_regions])
    df_de_age_all = pd.concat([df_de_age_all, df_2020_de_all])
    df_de_age_male = pd.concat([df_de_age_male, df_2020_de_males])
    df_de_age_female = pd.concat([df_de_age_female, df_2020_de_females])
    df_de_region = pd.concat([df_de_region, df_2020_de_regions])

    if verbose:
        df_2020_cv_all.info()
        print(df_2020_cv_all.head(20))
        df_2020_cv_males.info()
        print(df_2020_cv_males.head(20))
        df_2020_cv_females.info()
        print(df_2020_cv_females.head(20))
        df_2020_cv_regions.info()
        print(df_2020_cv_regions.head(20))

        df_2020_de_all.info()
        print(df_2020_de_all.head(20))
        df_2020_de_males.info()
        print(df_2020_de_males.head(20))
        df_2020_de_females.info()
        print(df_2020_de_females.head(20))
        df_2020_de_regions.info()
        print(df_2020_de_regions.head(20))

        df_vacs.info()
        df_cv19_age_all.info()
        df_de_age_all.info()
        df_cv19_age_male.info()
        df_de_age_male.info()
        df_cv19_age_female.info()
        df_de_age_female.info()
        df_cv19_region.info()
        df_de_region.info()

    return [df_vacs, df_cv19_age_all, df_de_age_all, df_cv19_age_male, df_de_age_male, df_cv19_age_female, 
            df_de_age_female, df_cv19_region, df_de_region]

def main():
    dfs = import_files(verbose=True)

if __name__ == '__main__':
    main()