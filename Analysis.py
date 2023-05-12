import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from collections import defaultdict
from pywaffle import Waffle
import numpy as np
import seaborn as sns
from plotly import graph_objects as go
import json
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
        other_dir = r"C:\Users\yblad\Documents\For Bsc\Year 3\Data Vis\Assessment\Data\Other"
    elif os.name == 'posix':
        deaths_dir = r"/mnt/c/Documents and Settings/yblad/Documents/For Bsc/Year 3/Data Vis/Assessment/Data/Deaths registered by year"
        vaccine_dir = r"/mnt/c/Documents and Settings/yblad/Documents/For Bsc/Year 3/Data Vis/Assessment/Data/Vaccination"
        other_dir = r"/mnt/c/Documents and Settings/yblad/Documents/For Bsc/Year 3/Data Vis/Assessment/Data/Other"

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
    vac_2022_23 = [x for x in files_vac if "2023" in x]
    for i in range(3, 9):
        df = import_excel_sheets(i, 4, vac_2022_23)[0]
        df['sheet_origin'] = i - 2
        dfs.append(df)

    df_vacs22_23 = pd.concat(dfs, axis=0, sort=False)
    df_vacs22_23['Month'] = pd.to_datetime(df_vacs22_23['Month'], format='%b-%y' , errors='raise')
    df_vacs22_23.dropna(subset=['Month'], inplace=True)


    #For the 2021 set we have to match the sheet_origin index of the 22/23
    #set, but there are no 3rd vaccine sheets. These occur at sheets 1 and 4.
    #Therefore we manually count skipping the missing sheets.
    #Both sets now have aligned 1...n sheet indices.
    dfs = []
    vac_2021 = [x for x in files_vac if "2021" in x]
    j = 1
    for i in range(1, 5):
        j += 1
        df = import_excel_sheets(i, 1, vac_2021)[0]
        if j % 4 == 0:
            j += 1
            df['sheet_origin'] = j
        else:
            df['sheet_origin'] = j

        dfs.append(df)

    df_vacs21 = pd.concat(dfs, axis=0, sort=False)

    df_vacs21.rename(columns={
        'Number of people who have received two vaccinations':'Number of people who had received two vaccinations',
        'Percentage of people who have received two vaccinations (%)':'Percentage of people who had received two vaccinations (%)',
        'Age standardised percentage of people who have received two vaccinations (%)':'Age standardised percentage of people who had received two vaccinations (%)',
        'Number of people who have not received a vaccination':'Number of people who had not received a vaccination',
        'Percentage of people who have not received a vaccination (%)':'Percentage of people who had not received a vaccination (%)',
        'Age standardised percentage of people who have not received a vaccination (%)':'Age standardised percentage of people who had not received a vaccination (%)'
    }, inplace=True)

    #Deal with notes at end of file and change string to proper datetime
    #df_vacs21['Month'] = pd.to_datetime(df_vacs21['Month'], format='mixed', errors='coerce')
    #df_vacs21['Month'] = df['Month'].apply(lambda x: datetime.datetime.strptime(x,'%b-Y').strftime('%b-%y'))
    #df_vacs21.dropna(subset=['Month'], inplace=True)

    df_vacs = pd.concat((df_vacs22_23, df_vacs21), axis=0, sort=False, ignore_index=True)

    #Handle covid deaths by age and by gender
    
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
    df_2020 = pd.read_excel(*files_deaths_pre22, sheet_name=6, skiprows=4, header=1, nrows=80)
    df_2020.dropna(how='all', inplace=True)
    df_2020.drop(columns=['Week ended','1 to 53'], inplace=True)

    df_2020_cv_all = df_2020[3:23]
    df_2020_cv_all.set_index('Unnamed: 1', inplace=True)
    df_2020_cv_all = df_2020_cv_all.T
    df_2020_cv_all.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_cv_all.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_cv_males = df_2020[25:45]
    df_2020_cv_males.set_index('Unnamed: 1', inplace=True)
    df_2020_cv_males = df_2020_cv_males.T
    df_2020_cv_males.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_cv_males.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_cv_females = df_2020[47:67]
    df_2020_cv_females.set_index('Unnamed: 1', inplace=True)
    df_2020_cv_females = df_2020_cv_females.T
    df_2020_cv_females.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_cv_females.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_cv_regions = df_2020[68:78]
    df_2020_cv_regions.set_index('Unnamed: 1', inplace=True)
    df_2020_cv_regions = df_2020_cv_regions.T
    df_2020_cv_regions.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_cv_regions.rename(columns={'East':'East of England'}, inplace=True)

    #All registered deaths. This time we need both files

    files_deaths_pre22 = get_files("Pre22", deaths_dir)
    files_deaths_pre22 = [x for x in files_deaths_pre22 if x.endswith('.xlsx')]
    f2021, f2020 = files_deaths_pre22[0], files_deaths_pre22[1]

    df_2020 = pd.read_excel(f2020, sheet_name=4, skiprows=4, header=1, nrows=92)
    df_2020.dropna(how='all', inplace=True)
    df_2020.drop(columns='Week ended', inplace=True)

    df_2020_de_all = df_2020[13:33]
    df_2020_de_all.set_index('Unnamed: 1', inplace=True)
    df_2020_de_all = df_2020_de_all.T
    df_2020_de_all.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_de_all.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_de_males = df_2020[35:55]
    df_2020_de_males.set_index('Unnamed: 1', inplace=True)
    df_2020_de_males = df_2020_de_males.T
    df_2020_de_males.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_de_males.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_de_females = df_2020[57:77]
    df_2020_de_females.set_index('Unnamed: 1', inplace=True)
    df_2020_de_females = df_2020_de_females.T
    df_2020_de_females.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_de_females.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2020_de_regions = df_2020[78:88]
    df_2020_de_regions.set_index('Unnamed: 1', inplace=True)
    df_2020_de_regions = df_2020_de_regions.T
    df_2020_de_regions.reset_index(drop=False, inplace=True, names='Week ending')
    df_2020_de_regions.rename(columns={'East':'East of England'}, inplace=True)

    df_2021 = pd.read_excel(f2021, sheet_name=4, skiprows=4, header=1, nrows=96)
    df_2021.dropna(how='all', inplace=True)
    df_2021.drop(columns='Week ended', inplace=True)
    df_2021 = df_2021.iloc[:,:-1]

    df_2021_de_all = df_2021[7:27]
    df_2021_de_all.set_index('Unnamed: 1', inplace=True)
    df_2021_de_all = df_2021_de_all.T
    df_2021_de_all.reset_index(drop=False, inplace=True, names='Week ending')
    df_2021_de_all.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2021_de_males = df_2021[29:49]
    df_2021_de_males.set_index('Unnamed: 1', inplace=True)
    df_2021_de_males = df_2021_de_males.T
    df_2021_de_males.reset_index(drop=False, inplace=True, names='Week ending')
    df_2021_de_males.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2021_de_females = df_2021[51:71]
    df_2021_de_females.set_index('Unnamed: 1', inplace=True)
    df_2021_de_females = df_2021_de_females.T
    df_2021_de_females.reset_index(drop=False, inplace=True, names='Week ending')
    df_2021_de_females.rename(columns={'1-4':'01-04','5-9':'05-09'}, inplace=True)

    df_2021_de_regions = df_2021[72:82]
    df_2021_de_regions.set_index('Unnamed: 1', inplace=True)
    df_2021_de_regions = df_2021_de_regions.T
    df_2021_de_regions.reset_index(drop=False, inplace=True, names='Week ending')
    df_2021_de_regions.rename(columns={'East':'East of England'}, inplace=True)

    #Merge 2020 frames with newer format frames
    #Set name attrs

    df_cv19_age_all = pd.concat([df_cv19_age_all, df_2020_cv_all])
    df_cv19_age_all.attrs['name'] = 'df_cv19_age_all'
    df_cv19_age_male = pd.concat([df_cv19_age_male, df_2020_cv_males])
    df_cv19_age_male.attrs['name'] = 'df_cv19_age_male'
    df_cv19_age_female = pd.concat([df_cv19_age_female, df_2020_cv_females])
    df_cv19_age_female.attrs['name'] = 'df_cv19_age_female'
    df_cv19_region = pd.concat([df_cv19_region, df_2020_cv_regions])
    df_cv19_region.attrs['name'] = 'df_cv19_region'
    df_de_age_all = pd.concat([df_de_age_all, df_2021_de_all, df_2020_de_all])
    df_de_age_all.attrs['name'] = 'df_de_age_all'
    df_de_age_male = pd.concat([df_de_age_male, df_2021_de_males, df_2020_de_males])
    df_de_age_male.attrs['name'] = 'df_de_age_male'
    df_de_age_female = pd.concat([df_de_age_female, df_2021_de_females, df_2020_de_females])
    df_de_age_female.attrs['name'] = 'df_de_age_female'

    df_de_region = pd.concat([df_de_region, df_2021_de_regions, df_2020_de_regions])
    df_de_region.attrs['name'] = 'df_de_region'

    df_vacs.attrs['name'] = 'df_vacs'

    #Remove duplicates. Due to the files naming stucture and how we have then appended
    #files, the newest version of the data is on top. Therefore, if we keep the first.
    #This ensures we capture any corrections made in subsequent issues. Not required
    #for vaccine data as this is pulled from a single issue.

    df_cv19_age_all.drop_duplicates(subset='Week ending', keep='first', inplace=True)
    df_cv19_age_male.drop_duplicates(subset='Week ending', keep='first', inplace=True)
    df_cv19_age_female.drop_duplicates(subset='Week ending', keep='first', inplace=True)
    df_cv19_region.drop_duplicates(subset='Week ending', keep='first', inplace=True)
    df_de_age_all.drop_duplicates(subset='Week ending', keep='first', inplace=True)
    df_de_age_male.drop_duplicates(subset='Week ending', keep='first', inplace=True)
    df_de_age_female.drop_duplicates(subset='Week ending', keep='first', inplace=True)
    df_de_region.drop_duplicates(subset='Week ending', keep='first', inplace=True)

    df_cv19_age_male['gender'] = 'male'
    df_cv19_age_female['gender'] = 'female'
    df_cv19_by_gender = pd.concat((df_cv19_age_male, df_cv19_age_female))
    df_cv19_by_gender.attrs['name'] = 'df_cv19_by_gender'

    df_de_age_male['gender'] = 'male'
    df_de_age_female['gender'] = 'female'
    df_de_by_gender = pd.concat((df_de_age_male, df_de_age_female))
    df_de_by_gender.attrs['name'] = 'df_de_by_gender'

    #Get population by age. This comes on a per city per year of age basis, so we need to re-bin it
    #This is left to the analysis calls as we will bin differently at different stages
    if os.name == 'nt':
        df_pop_age = pd.read_excel(other_dir + "\Population by age.xlsx", sheet_name=0)
    elif os.name == 'posix':
        df_pop_age = pd.read_excel(other_dir + "/Population by age.xlsx", sheet_name=0)

    #Get regional population by (ITL1) upper region
    if os.name == 'nt':
        df_pop_region = pd.read_excel(other_dir + "\Population by region.xlsx", sheet_name=1)
    elif os.name == 'posix':
        df_pop_region = pd.read_excel(other_dir + "/Population by region.xlsx", sheet_name=1)

    df_pop_region.attrs['name'] = 'df_pop_region'
    df_pop_age.attrs['name'] ='df_pop_age'

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

        df_2021_de_all.info()
        print(df_2021_de_all.head(20))
        df_2021_de_males.info()
        print(df_2021_de_males.head(20))
        df_2021_de_females.info()
        print(df_2021_de_females.head(20))
        df_2021_de_regions.info()
        print(df_2021_de_regions.head(20))

        df_vacs.info()
        df_cv19_age_all.info()
        df_de_age_all.info()
        df_cv19_age_male.info()
        df_de_age_male.info()
        df_cv19_age_female.info()
        df_de_age_female.info()
        df_cv19_region.info()
        df_de_region.info()

    return [df_vacs, df_cv19_age_all, df_de_age_all, df_cv19_by_gender, df_de_by_gender, df_cv19_region, df_de_region, df_pop_age, df_pop_region]

def get_vac_slices(df_vacs):
    '''Unpack the concatenated vaccination dataframe into individual frames for each type of data.'''

    df_vacs = df_vacs.copy()

    df_18_3_vacs = df_vacs.loc[df_vacs['sheet_origin'] == 1]
    df_18_2_vacs = df_vacs.loc[df_vacs['sheet_origin'] == 2]
    df_18_0_vacs = df_vacs.loc[df_vacs['sheet_origin'] == 3]
    df_50_3_vacs = df_vacs.loc[df_vacs['sheet_origin'] == 4]
    df_50_2_vacs = df_vacs.loc[df_vacs['sheet_origin'] == 5]
    df_50_0_vacs = df_vacs.loc[df_vacs['sheet_origin'] == 6]

    df_18_3_vacs.attrs['name'] = "18+ three vaccines"
    df_18_2_vacs.attrs['name'] = "18+ two vaccines"
    df_18_0_vacs.attrs['name'] = "18+ no vaccines"
    df_50_3_vacs.attrs['name'] = "50+ three vaccines"
    df_50_2_vacs.attrs['name'] = "50+ two vaccines"
    df_50_0_vacs.attrs['name'] = "50+ no vaccines"

    #The older style data put sex in category rather than a seperate sex column. Move it across.

    df_18_3_vacs.loc[:,'Sex'] = df_18_3_vacs.apply(lambda x: x.Sex if (x['Category type'] != 'Sex') else x.Category, axis=1)
    df_18_2_vacs.loc[:,'Sex'] = df_18_2_vacs.apply(lambda x: x.Sex if (x['Category type'] != 'Sex') else x.Category, axis=1)
    df_18_0_vacs.loc[:,'Sex'] = df_18_0_vacs.apply(lambda x: x.Sex if (x['Category type'] != 'Sex') else x.Category, axis=1)
    df_50_3_vacs.loc[:,'Sex'] = df_50_3_vacs.apply(lambda x: x.Sex if (x['Category type'] != 'Sex') else x.Category, axis=1)
    df_50_2_vacs.loc[:,'Sex'] = df_50_2_vacs.apply(lambda x: x.Sex if (x['Category type'] != 'Sex') else x.Category, axis=1)
    df_50_0_vacs.loc[:,'Sex'] = df_50_0_vacs.apply(lambda x: x.Sex if (x['Category type'] != 'Sex') else x.Category, axis=1)

    df_18_3_vacs['Sex'] = df_18_3_vacs['Sex'].fillna('Persons')
    df_18_2_vacs['Sex'] = df_18_2_vacs['Sex'].fillna('Persons')
    df_18_0_vacs['Sex'] = df_18_0_vacs['Sex'].fillna('Persons')
    df_50_3_vacs['Sex'] = df_50_3_vacs['Sex'].fillna('Persons')
    df_50_2_vacs['Sex'] = df_50_2_vacs['Sex'].fillna('Persons')
    df_50_0_vacs['Sex'] = df_50_0_vacs['Sex'].fillna('Persons')

    return df_18_3_vacs, df_18_2_vacs, df_18_0_vacs, df_50_3_vacs, df_50_2_vacs, df_50_0_vacs

def fill_pop_bins(x, cats, bins):
    '''Returns the bins for the associated matching category,
    or null string if not found'''
    for i, cat in enumerate(cats):
        if int(x) in cat:
            return bins[i]
        else:
            pass
    
    return ""

def get_df(dfs, df_name):
    '''Convenience function. Returns the named df from the list of dfs.
    Relies on the fact we have previous set the df.attrs['name'] attribute.
    
    This would be more efficient if it were refactored such that the dfs are in a
    dict which we can query, but as the list is short this is fast enough.'''

    df = [x for x in dfs if x.attrs['name'] == df_name]

    if len(df) == 1:
        return df[0]
    elif len(df) > 1:
        raise ValueError('Multiple dataframes found with the given name attribute')
    elif len(df) == 0:
        raise ValueError('No dataframes found with the given name attribute')

def deaths_analysis(dfs):
    #Get required data and reformat into into 'long' form for seaborn/matplotlib

    df_cv19_all = get_df(dfs, 'df_cv19_age_all').drop(columns=['Week number']).sort_values(by='Week ending')
    df_cv19_gender = get_df(dfs, 'df_cv19_by_gender').drop(columns=['Week number']).sort_values(by='Week ending')
    df_cv19_region = get_df(dfs, 'df_cv19_region').drop(columns=['Week number', 'Wales']).sort_values(by='Week ending')
    df_de_all = get_df(dfs, 'df_de_age_all').drop(columns=['Week number']).sort_values(by='Week ending') 
    df_de_gender = get_df(dfs, 'df_de_by_gender').drop(columns=['Week number','All ages']).sort_values(by='Week ending')
    df_age_pop = get_df(dfs, 'df_pop_age')
    df_region_pop = get_df(dfs, 'df_pop_region')

    #Prepare the population by age data. 
    #We group by age first to collapse cities into an all England figure,
    #then group by age bin in line with our deaths data

    df_age_pop = df_age_pop.groupby(['Age (101 categories) Code'])['Observation'].agg('sum').reset_index()

    categories = [x for x in df_cv19_all.columns if x != 'Week ending' and x!= 'All ages']
    bins = categories.copy()

    #Expand hypenated ranges
    #e.g 1-5 -> 1,2,3,4,5

    categories[0] = "0-1"
    categories[-1] = "90-100"

    for i, cat in enumerate(categories):
        split = cat.split('-')
        categories[i] = list(range(int(split[0]), int(split[1]) + 1))

    #test whether age is in each category and assign the assiociated bin
    #then group by bin

    df_age_pop['bin'] = df_age_pop['Age (101 categories) Code'].apply(fill_pop_bins, args=(categories, bins))
    df_age_pop = df_age_pop.groupby(['bin'])['Observation'].agg('sum')

    #Get standardisation factor. This is just the fraction of population in each bin
    total = df_age_pop.sum()
    df_age_pop = df_age_pop.to_frame()
    df_age_pop['factor'] = df_age_pop / total

    df_age_pop = df_age_pop.reset_index().pivot(columns='bin', values='factor')

    #For the region data we simply need to calculate the factor
    total = df_region_pop.iloc[:, 1].sum()
    df_region_pop['factor'] = df_region_pop.iloc[:, 1] / total

    df_region_pop = df_region_pop.reset_index().pivot(columns='Population', values='factor')

    sns.set_theme()

    #Proportion of c19 cases in each age bracket over time
    categories = [x for x in df_cv19_all.columns if x != 'Week ending' and x!= 'All ages']
    plt.stackplot(df_cv19_all['Week ending'], df_cv19_all[categories].T, labels=categories)
    plt.title("Covid 19 deaths by age group")
    plt.xlabel("Week ending")
    plt.xticks(rotation=90)
    plt.ylabel("Deaths")
    plt.legend()
    plt.show()

    #get normalised data by dividing each datum b the total for that date
    normalised = df_cv19_all[categories].divide(df_cv19_all[categories].sum(axis=1), axis=0)
    plt.stackplot(df_cv19_all['Week ending'], normalised.T, labels=categories)
    plt.title("Covid proportion of 19 deaths by age group")
    plt.xlabel("Week ending")
    plt.xticks(rotation=90)
    plt.ylabel("Deaths (normalised)")
    plt.legend()
    plt.show()

    #As above, but by gender
    #We need to stack the male/female totals side by side, then normalise
    data = 'All ages'
    df_cv19_gender.fillna(0, inplace=True)
    male, female = df_cv19_gender.query('gender == "male"'), df_cv19_gender.query('gender == "female"')
    male, female = male[['Week ending', data]], female[['Week ending', data]]
    normalised = pd.concat((male, female), axis=1)
    normalised = normalised[data].divide(normalised[data].sum(axis=1), axis=0)

    plt.stackplot(df_cv19_gender['Week ending'].drop_duplicates(), normalised.T, labels=['male','female'])
    plt.axhline(y=0.5, ls='--', color='black', label="0.5")
    plt.title("Covid proportion of deaths by gender")
    plt.ylabel("Deaths (normalised)")
    plt.xlabel("Week ending")
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

    #same for region
    df_cv19_region.fillna(0, inplace=True)
    categories = [x for x in df_cv19_region.columns if x != 'Week ending']
    normalised = df_cv19_region[categories].divide(df_cv19_region[categories].sum(axis=1), axis=0)
    plt.stackplot(df_cv19_region['Week ending'], normalised.T, labels=categories)
    plt.title("Covid proportion of 19 deaths by region")
    plt.xlabel("Week ending")
    plt.xticks(rotation=90)
    plt.ylabel("Deaths (normalised)")
    plt.legend()
    plt.show()

    #bumb chart for region
    #we get the rank of each region by deaths and then plot on the ranks
    df_cv19_region['rank'] = df_cv19_region[categories].apply(rankdata, axis=1)
    for i, col in enumerate(categories):
        plt.plot(df_cv19_region['Week ending'], np.stack(df_cv19_region['rank'].to_numpy(), axis=0)[:,i], label=col)

    plt.title("Regional rank of deaths due to covid 19")
    plt.xlabel("Week ending")
    plt.xticks(rotation=90)
    plt.ylabel("Rank")
    plt.legend()
    plt.show()

    #Waffle charts of proportion of covid deaths for each age group
    #over whole time period. Create a list with proportion of cv19 deaths

    categories = [x for x in df_cv19_all.columns if x != 'Week ending' and x != 'All ages']
    df_cv19_prop = pd.DataFrame()
    df_cv19_all.set_index('Week ending', inplace=True)
    df_de_all.set_index('Week ending', inplace=True)
    totals = []

    for col in categories:
        df_cv19_prop[col] = df_cv19_all[col].divide(df_de_all[col])
        df_cv19_prop[col] = df_cv19_prop[col].where(df_cv19_prop[col] != np.inf, 0)
        totals.append(df_cv19_prop[col].sum() / len(df_cv19_all[col]))

    #Now turn into 2d array of [proportion cv19_deaths, 1- proportion cv19_deaths]

    totals = np.asarray(totals).reshape(-1,1)
    totals = np.append(totals, 1 - totals, axis=1)

    #Pywaffle is essentially a wrapper for matplotlib to make waffle charts
    #populate sublot dictionary with following dict format
    #see https://pywaffle.readthedocs.io/en/latest/examples/subplots.html
               # (row,column,index): {
                #'values': totals*100,
                #'labels': ['Covid19', 'All deaths'],
                #'legend': {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 8},
                #'title': {'label': 'category name', 'loc': 'left', 'fontsize': 12}
            #               }

    plots = defaultdict(dict)

    for i in range(len(totals)):
        plots[(4,5,i+1)]['values'] = totals[i]*100
        plots[(4,5,i+1)]['labels'] = ['Covid19', 'All deaths']
        plots[(4,5,i+1)]['legend'] = {'loc': 'upper left', 'bbox_to_anchor': (1.05, 1), 'fontsize': 8}
        plots[(4,5,i+1)]['title'] = {'label': categories[i], 'loc': 'left', 'fontsize': 12}

    fig = plt.figure(
        FigureClass=Waffle,
        plots=plots,
        rows=5,
        columns=10,
        cmap_name="Accent", 
        rounding_rule='nearest', 
        figsize=(10, 10)
    )

    fig.suptitle('Proportion of deaths to covid 19', fontsize=14, fontweight='bold')
    fig.supxlabel('1 block = 2% of deaths', fontsize=8, ha='right')
    plt.show()

    #Lineplot of proportion of deaths over time for each group

    for i , col in enumerate(categories):
        plt.plot(df_cv19_all.index, df_cv19_prop[col], label = col)

    plt.title("Proportion of covid deaths per age group")
    plt.xlabel("Week ending")
    plt.xticks(rotation=90)
    plt.ylabel("% of deaths")
    plt.legend()
    plt.show()

    #Lineplot proportion of deaths over time all combined
    df_de_all['All ages'] = df_de_all['All ages'].where(df_de_all['All ages'].isnull() == False, df_de_all[categories].sum(axis=1))
    df_cv19_all['All ages'] = df_cv19_all['All ages'].where(df_cv19_all['All ages'].isnull() == False, df_cv19_all[categories].sum(axis=1))
    df_cv19_prop['all'] = df_cv19_all['All ages'].divide(df_de_all['All ages'])
    df_cv19_prop['all'] = df_cv19_prop['all'].where(df_cv19_prop['all'] != np.inf, 0)
    plt.plot(df_cv19_all.index, df_cv19_prop['all'], label='All data')

    plt.title("Proportion of covid deaths in all age groups")
    plt.xlabel("Week ending")
    plt.xticks(rotation=90)
    plt.ylabel("% of deaths")
    plt.show()

    #Rerun of some of the above with addition of population standardisation
    df_cv19_all.reset_index(inplace=True)
    categories = [x for x in df_cv19_all.columns if x != 'Week ending' and x!= 'All ages']
    
    for col in categories:
        df_cv19_all[col] = df_cv19_all[col] * df_age_pop[col].dropna().squeeze()

    normalised = df_cv19_all[categories].divide(df_cv19_all[categories].sum(axis=1), axis=0)
    plt.stackplot(df_cv19_all['Week ending'], normalised.T, labels=categories)
    plt.title("Covid proportion of 19 deaths by age group, population standardised")
    plt.xlabel("Week ending")
    plt.xticks(rotation=90)
    plt.ylabel("Deaths (normalised)")
    plt.legend()
    plt.show()

    #same for region
    categories = [x for x in df_cv19_region.columns if x != 'Week ending' and x != 'rank']
    for col in categories:
        df_cv19_region[col] = df_cv19_region[col] * df_region_pop[col].dropna().squeeze()

    normalised = df_cv19_region[categories].divide(df_cv19_region[categories].sum(axis=1), axis=0)
    plt.stackplot(df_cv19_region['Week ending'], normalised.T, labels=categories)
    plt.title("Covid proportion of 19 deaths by region, population standardised")
    plt.xlabel("Week ending")
    plt.xticks(rotation=90)
    plt.ylabel("Deaths (normalised)")
    plt.legend()
    plt.show()

    #Using the population standardised data, create choropleth map of total deaths p/region
    #We normalise the data again to make the scale make sense, but this time we use a different method
    #as we are getting the normalised total rather than the normalised figure per week
    #We multiply be 100 to make the scale the same on all choropleths (% rather than decimal)

    with open('nuts1.json') as f:
        geo_uk = json.load(f)

    z = df_cv19_region[categories].sum(axis=0).divide(df_cv19_region[categories].sum(axis=0).sum()) * 100

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson = geo_uk,
            featureidkey = "properties.NUTS112NM", 
            locations = categories, 
            z = z, 
            zauto = True,
            colorscale = 'rdbu',
            showscale = True,
            reversescale= True
        )
    )

    fig.update_layout(
        mapbox_style = "carto-positron", 
        mapbox_zoom = 5.5, 
        mapbox_center = {"lat": 52.53, "lon": 1.77},
        title = "Deaths by region (population standardised)"
    )

    fig.show()

def vacs_analysis(dfs):
    df_vacs = get_df(dfs, 'df_vacs')

    sns.set_theme()

    #replace NaNs and characters indicaters for suppressed values
    df_vacs[['Category type','Category']] = df_vacs[['Category type','Category']].fillna("Total")
    df_vacs.fillna(0, inplace=True)
    df_vacs.replace("x", 0, inplace=True)
    df_vacs.replace("c", 0, inplace=True)

    #monthly vaccine rates (all)

    df_18_3_vacs, df_18_2_vacs, df_18_0_vacs, df_50_3_vacs, df_50_2_vacs, df_50_0_vacs = get_vac_slices(df_vacs)

    dfs = [df_18_3_vacs.copy(), df_18_2_vacs.copy(), df_18_0_vacs.copy(), df_50_3_vacs.copy(), df_50_2_vacs.copy(), 
           df_50_0_vacs.copy()]

    #Set filter

    for i, df in enumerate(dfs):
        dfs[i] = df.loc[(df['Category'] == 'Total') & (df['Sub-category'] == 'England') & (df['Sex'] == 'Persons')]

    #remove duplicates

    for i, df in enumerate(dfs):
        dfs[i] = df.drop_duplicates(subset='Month', keep='first').sort_values(by='Month')

    #We need to pull from a different column at different iterations
    for i, df in enumerate(dfs):
        if i == 0 or i == 3:
            plt.plot(df['Month'], df['Percentage of people who had received three vaccinations (%)'],
                     label=f"{df.attrs['name']} %")
        elif i == 1 or i == 4:
            plt.plot(df['Month'], df['Percentage of people who had received two vaccinations (%)'],
                     label=f"{df.attrs['name']} %")
        elif i == 2 or i == 5:
            plt.plot(df['Month'], df['Percentage of people who had not received a vaccination (%)'],
                     label=f"{df.attrs['name']} %")


    plt.title("Proportion of vaccinated population")
    plt.xlabel("Month")
    plt.xticks(rotation=90)
    plt.ylabel("% vaccinated")
    plt.legend()
    plt.show()

    #By gender, male then female

#    df_18_3_vacs, df_18_2_vacs, df_18_0_vacs, df_50_3_vacs, df_50_2_vacs, df_50_0_vacs = get_vac_slices(df_vacs)

    dfs = [df_18_3_vacs.copy(), df_18_2_vacs.copy(), df_18_0_vacs.copy(), df_50_3_vacs.copy(), df_50_2_vacs.copy(), 
           df_50_0_vacs.copy()]

    #Set filter

    for i, df in enumerate(dfs):
        dfs[i] = df.loc[((df['Category'] == 'Total') | (df['Category'] == 'Sex')) & (df['Sub-category'] == 'England') & (df['Sex'] == 'Male')]

    #remove duplicates
    for df in dfs:
        df = df.drop_duplicates(subset='Month', keep='first').sort_values(by='Month')

    for i, df in enumerate(dfs):
        if i == 0 or i == 3:
            plt.plot(df['Month'], df['Percentage of people who had received three vaccinations (%)'],
                     label=f"{df.attrs['name']} %")
        elif i == 1 or i == 4:
            plt.plot(df['Month'], df['Percentage of people who had received two vaccinations (%)'],
                     label=f"{df.attrs['name']} %")
        elif i == 2 or i == 5:
            plt.plot(df['Month'], df['Percentage of people who had not received a vaccination (%)'],
                     label=f"{df.attrs['name']} %")


    plt.title("Proportion of vaccinated population (males)")
    plt.xlabel("Month")
    plt.xticks(rotation=90)
    plt.ylabel("% vaccinated")
    plt.legend()
    plt.show()

    dfs = [df_18_3_vacs.copy(), df_18_2_vacs.copy(), df_18_0_vacs.copy(), df_50_3_vacs.copy(), df_50_2_vacs.copy(), 
           df_50_0_vacs.copy()]

    #Set filter

    for i, df in enumerate(dfs):
        dfs[i] = df.loc[((df['Category'] == 'Total') | (df['Category'] == 'Sex')) & (df['Sub-category'] == 'England') & (df['Sex'] == 'Female')]

    #remove duplicates
    for df in dfs:
        df = df.drop_duplicates(subset='Month', keep='first').sort_values(by='Month')

    for i, df in enumerate(dfs):
        if i == 0 or i == 3:
            plt.plot(df['Month'], df['Percentage of people who had received three vaccinations (%)'],
                     label=f"{df.attrs['name']} %")
        elif i == 1 or i == 4:
            plt.plot(df['Month'], df['Percentage of people who had received two vaccinations (%)'],
                     label=f"{df.attrs['name']} %")
        elif i == 2 or i == 5:
            plt.plot(df['Month'], df['Percentage of people who had not received a vaccination (%)'],
                     label=f"{df.attrs['name']} %")


    plt.title("Proportion of vaccinated population (Females)")
    plt.xlabel("Month")
    plt.xticks(rotation=90)
    plt.ylabel("% vaccinated")
    plt.legend()
    plt.show()

    #Regional data. This time, for ease of comparison, we will look at just the
    #percentage triple vaccinated. This is the 18+ tripel vac df

    triple_vac = df_18_3_vacs.copy()

    #Set filter

    regions = ['East Midlands',
               'East of England', 'London', 'North East', 'North West',
               'South East', 'South West', 'West Midlands',
               'Yorkshire and The Humber'
                ]

    triple_vac = triple_vac.loc[triple_vac['Category'] == 'Total']
    triple_vac = triple_vac.loc[triple_vac['Sub-category'].apply(lambda x: x in regions)]

    dfs = []

    for i in range(len(regions)):
        dfs.append(triple_vac.loc[triple_vac['Sub-category'] == regions[i]])
        dfs[i].attrs['name'] = regions[i]
        dfs[i] = dfs[i].drop_duplicates(subset='Month', keep='first').sort_values(by='Month')

    for df in dfs:
        plt.plot(df['Month'], df['Percentage of people who had received three vaccinations (%)'],
                    label=f"{df.attrs['name']} %")
        #plt.fill_between(df['Month'], df['95% confidence interval - Lower bound'], df['95% confidence interval - Upper bound'])

    plt.title("Proportion of fully vaccinated population (by region)")
    plt.xlabel("Month")
    plt.xticks(rotation=90)
    plt.ylabel("% triple vaccinated")
    plt.legend()
    plt.show()

    #regional chorpleth of triple vaccine rate
    with open('nuts1.json') as f:
        geo_uk = json.load(f)

    z = triple_vac.groupby('Sub-category')['Percentage of people who had received three vaccinations (%)'].agg('max')

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson = geo_uk,
            featureidkey = "properties.NUTS112NM",
            locations = triple_vac['Sub-category'],
            z = z,
            zauto = True,
            colorscale = 'rdbu',
            showscale = True,
        )
    )

    fig.update_layout(
        mapbox_style = "carto-positron",
        mapbox_zoom = 5.5,
        mapbox_center = {"lat": 52.53, "lon": 1.77},
        title = "Fully vaccinated population by region (all adults)"
    )

    fig.show()

    #As above, but for 50+ exclusivley

    triple_vac = df_50_3_vacs.copy()

    #Set filter

    regions = ['East Midlands',
               'East of England', 'London', 'North East', 'North West',
               'South East', 'South West', 'West Midlands',
               'Yorkshire and The Humber'
                ]

    triple_vac = triple_vac.loc[triple_vac['Category'] == 'Total']
    triple_vac = triple_vac.loc[triple_vac['Sub-category'].apply(lambda x: x in regions)]

    dfs = []

    for i in range(len(regions)):
        dfs.append(triple_vac.loc[triple_vac['Sub-category'] == regions[i]])
        dfs[i].attrs['name'] = regions[i]
        dfs[i] = dfs[i].drop_duplicates(subset='Month', keep='first').sort_values(by='Month')

    for df in dfs:
        plt.plot(df['Month'], df['Percentage of people who had received three vaccinations (%)'],
                    label=f"{df.attrs['name']} %")
        #plt.fill_between(df['Month'], df['95% confidence interval - Lower bound'], df['95% confidence interval - Upper bound'])

    plt.title("Proportion of fully vaccinated population, 50+, (by region)")
    plt.xlabel("Month")
    plt.xticks(rotation=90)
    plt.ylabel("% triple vaccinated")
    plt.legend()
    plt.show()

    #regional chorpleth of triple vaccine rate (50+)

    z = triple_vac.groupby('Sub-category')['Percentage of people who had received three vaccinations (%)'].agg('max')

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson = geo_uk,
            featureidkey = "properties.NUTS112NM", 
            locations = triple_vac['Sub-category'], 
            z = z, 
            zauto = True,
            colorscale = 'rdbu',
            showscale = True,
        )
    )

    fig.update_layout(
        mapbox_style = "carto-positron", 
        mapbox_zoom = 5.5, 
        mapbox_center = {"lat": 52.53, "lon": 1.77},
        title = "Fully vaccinated population by region (50+ years old)"
    )

    fig.show()


def deaths_vacs_analysis(dfs):
    df_cv19_all = get_df(dfs, 'df_cv19_age_all').drop(columns=['Week number']).sort_values(by='Week ending')
    df_cv19_region = get_df(dfs, 'df_cv19_region').drop(columns=['Week number', 'Wales']).sort_values(by='Week ending')
    df_age_pop = get_df(dfs, 'df_pop_age')
    df_region_pop = get_df(dfs, 'df_pop_region')
    df_vacs = get_df(dfs, 'df_vacs')

    sns.set_theme()

    #Prepare vaccines

    #replace NaNs and characters indicaters for suppressed values
    df_vacs[['Category type','Category']] = df_vacs[['Category type','Category']].fillna("Total")
    df_vacs.fillna(0, inplace=True)
    df_vacs.replace("x", 0, inplace=True)
    df_vacs.replace("c", 0, inplace=True)

    #monthly vaccine rates (all)

    df_18_3_vacs, df_18_2_vacs, df_18_0_vacs, df_50_3_vacs, df_50_2_vacs, df_50_0_vacs = get_vac_slices(df_vacs)

    dfs_vacs = [df_18_3_vacs.copy(), df_18_2_vacs.copy(), df_18_0_vacs.copy(), df_50_3_vacs.copy(), df_50_2_vacs.copy(), 
           df_50_0_vacs.copy()]
    dfs_vacs_all_adults = dfs_vacs[0:3]
    
    #Prepare the population by age data. 
    #We group by age first to collapse cities into an all England figure,
    #then group by age bin in line with our deaths data

    df_age_pop = df_age_pop.groupby(['Age (101 categories) Code'])['Observation'].agg('sum').reset_index()

    categories = [x for x in df_cv19_all.columns if x != 'Week ending' and x!= 'All ages']
    bins = categories.copy()

    #Expand hypenated ranges
    #e.g 1-5 -> 1,2,3,4,5

    categories[0] = "0-1"
    categories[-1] = "90-100"

    for i, cat in enumerate(categories):
        split = cat.split('-')
        categories[i] = list(range(int(split[0]), int(split[1]) + 1))

    #test whether age is in each category and assign the assiociated bin
    #then group by bin

    df_age_pop['bin'] = df_age_pop['Age (101 categories) Code'].apply(fill_pop_bins, args=(categories, bins))
    df_age_pop = df_age_pop.groupby(['bin'])['Observation'].agg('sum')

    #Get standardisation factor. This is just the fraction of population in each bin
    total = df_age_pop.sum()
    df_age_pop = df_age_pop.to_frame()
    df_age_pop['factor'] = df_age_pop / total

    df_age_pop = df_age_pop.reset_index().pivot(columns='bin', values='factor')

    #For the region data we simply need to calculate the factor
    total = df_region_pop.iloc[:, 1].sum()
    df_region_pop['factor'] = df_region_pop.iloc[:, 1] / total

    df_region_pop = df_region_pop.reset_index().pivot(columns='Population', values='factor')

    #Regroup cv19 deaths data by month, to align with vaccine data
    df_cv19_all.set_index('Week ending', inplace=True)
    categories = [x for x in df_cv19_all.columns if x != 'Week ending' and x!= 'All ages']
    df_cv19_all = df_cv19_all.groupby(pd.Grouper(freq='MS'))[categories].agg('sum')

    #population standardisation

    for col in categories:
        df_cv19_all[col] = df_cv19_all[col] * df_age_pop[col].dropna().squeeze()

    categories = [x for x in df_cv19_region.columns if x != 'Week ending' and x != 'rank']
    for col in categories:
        df_cv19_region[col] = df_cv19_region[col] * df_region_pop[col].dropna().squeeze()

    #Build new dataframes with features we want for correlation analysis

    categories = [x for x in df_cv19_all.columns if x != 'Week ending' and x!= 'All ages']

    df_cv19_age_vacs = pd.DataFrame()

    for col in categories:
        df_cv19_age_vacs[col] = df_cv19_all[col]

    #Set filter for vaccination data, and then add their features to the new dataframe

    for i, df in enumerate(dfs_vacs_all_adults):
        dfs_vacs_all_adults[i] = df.loc[(df['Category'] == 'Total') & (df['Sub-category'] == 'England') & (df['Sex'] == 'Persons')]

    for i, df in enumerate(dfs_vacs_all_adults):
        dfs_vacs_all_adults[i] = df.drop_duplicates(subset='Month', keep='first').sort_values(by='Month')

    allow_columns = ['Percentage of people who had not received a vaccination (%)',
                     'Percentage of people who had received two vaccinations (%)',
                    'Percentage of people who had received three vaccinations (%)'
    ]

    #drop unused columns from each, leaving only the one data column plus metadata.
    dfs_vacs_all_adults[0] = dfs_vacs_all_adults[0].drop(columns=['Percentage of people who had not received a vaccination (%)',
                                                                'Percentage of people who had received two vaccinations (%)',])
    dfs_vacs_all_adults[1] = dfs_vacs_all_adults[1].drop(columns=['Percentage of people who had not received a vaccination (%)',
                                                                'Percentage of people who had received three vaccinations (%)'])
    dfs_vacs_all_adults[2] = dfs_vacs_all_adults[2].drop(columns=['Percentage of people who had received two vaccinations (%)',
                                                            'Percentage of people who had received three vaccinations (%)'])

    #Add the data column from each vaccine df to the df we are building, ignoring all metadata

    for i, df in enumerate(dfs_vacs_all_adults):
        dfs_vacs_all_adults[i] = df.set_index('Month')

        for col in dfs_vacs_all_adults[i].columns:
            if col in allow_columns:
                df_cv19_age_vacs[col] = dfs_vacs_all_adults[i][col]

    exclude_columns = ['<1','01-04','05-09','10-14']
    corr_cols = [x for x in df_cv19_age_vacs.columns if x not in exclude_columns]
    df_age_corr = df_cv19_age_vacs[corr_cols]
    df_age_corr.rename(columns={'Percentage of people who had not received a vaccination (%)':'No Vaccine (%)',
                                  'Percentage of people who had received two vaccinations (%)':'Two Vaccines (%)',
                                  'Percentage of people who had received three vaccinations (%)':'Three Vaccines (%)'
                                }, inplace=True)
    df_age_corr = df_age_corr.corr(method='pearson')

    sns.heatmap(df_age_corr, annot=True)
    plt.show()

def main(FIRST_RUN):
    #import main files and serialise. 
    #If not first run, get pre-serialised data files

    if FIRST_RUN:
        dfs = import_files(verbose=False)

        for df in dfs:
            df.to_pickle(f"{df.attrs['name']}.pkl")

    else:
        dfs = []

        if os.name == 'nt':
            dir = r"C:\Users\yblad\Documents\For Bsc\Year 3\Data Vis"
        elif os.name == 'posix':
            dir = r"/mnt/c/Documents and Settings/yblad/Documents/For Bsc/Year 3/Data Vis"

        files = get_files('Assessment', dir)
        files = [x for x in files if ".pkl" in x]

        for f in files:
            dfs.append(pd.read_pickle(f))

    deaths_analysis(dfs)
    vacs_analysis(dfs)
    deaths_vacs_analysis(dfs)

if __name__ == '__main__':
    FIRST_RUN = False
    main(FIRST_RUN)