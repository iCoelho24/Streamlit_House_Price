import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, OrdinalEncoder

def artefato_deployment(df):
    
    df.drop(columns=['Id'], inplace=True)

    print('Iniciando tratamento de valores nulos.')

    if df['LotFrontage'].isnull().sum() > 0:
        df['LotFrontage'].fillna(70.04, inplace=True)


    if df['Alley'].isnull().sum() >0:
        df['Alley'].fillna('None', inplace= True)

    if df['MasVnrType'].isnull().sum() >0:
        df['MasVnrType'].fillna('None', inplace=True)

    if df['MasVnrArea'].isnull().sum() >0:
        df['MasVnrArea'].fillna(0, inplace=True)
    
    if df['BsmtQual'].isnull().sum() >0:
        df['BsmtQual'].fillna(0, inplace=True)

    if df['BsmtCond'].isnull().sum() >0:
        df['BsmtCond'].fillna(0, inplace=True)

    if df['BsmtExposure'].isnull().sum() >0:
        df['BsmtExposure'].fillna(0, inplace=True)

    if df['BsmtFinType1'].isnull().sum() >0:
        df['BsmtFinType1'].fillna(0, inplace=True)
        
    if df['BsmtFinType2'].isnull().sum() >0:
        df['BsmtFinType1'].fillna(0, inplace=True)

    if df['Electrical'].isnull().sum() >0:
        df['Electrical'].fillna('SBrkr', inplace=True)
        
    if df['FireplaceQu'].isnull().sum() > 0:
        df['FireplaceQu'].fillna('None', inplace=True)

    if df['GarageType'].isnull().sum() > 0:
        df['GarageType'].fillna('None', inplace=True)

    if df['GarageYrBlt'].isnull().sum() > 0:
        df['GarageYrBlt'].fillna('None', inplace=True)

    if df['GarageFinish'].isnull().sum() > 0:
        df['GarageFinish'].fillna('None', inplace=True)

    if df['GarageQual'].isnull().sum() > 0:
        df['GarageQual'].fillna('None', inplace=True)

    if df['GarageCond'].isnull().sum() > 0:
        df['GarageCond'].fillna('None', inplace=True)
    
    if df['PoolQC'].isnull().sum() > 0:
        df['PoolQC'].fillna('None', inplace=True)

    if df['Fence'].isnull().sum() > 0:
        df['Fence'].fillna('None', inplace=True)

    if df['MiscFeature'].isnull().sum() > 0:
        df['MiscFeature'].fillna('None', inplace=True)

    df.fillna(0, inplace=True)

    print('Tratamento de Nulos finalizados.')


    df.drop(columns=['GarageYrBlt', '1stFlrSF', '2ndFlrSF', 'MiscFeature' , 'PoolQC' , 'Heating' , 'Street' , 'Utilities' , 'RoofMatl' , 'Condition2', 'ScreenPorch', '3SsnPorch', 'PoolArea', 'MiscVal', 'BsmtHalfBath', 'LowQualFinSF'], inplace= True)

    print('Iniciando a normalização dos dados')

    l1_transf = ['LotArea', 'BsmtFinSF1', 'BsmtUnfSF',  'KitchenAbvGr', 'GarageCars']

    log2_tranf = ['MSSubClass', 'OverallCond', 'MasVnrArea', 'BsmtFinSF2', 'GrLivArea', 'BsmtFullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']

    col_categ_f = ['MSZoning', 'Alley', 'LotShape',
                   'LandContour', 'LotConfig', 'LandSlope', 
                   'Neighborhood','Condition1','BldgType',
                   'HouseStyle', 'RoofStyle',
                   'Exterior1st','Exterior2nd','MasVnrType',
                   'ExterQual','ExterCond','Foundation',
                   'BsmtQual','BsmtCond','BsmtExposure',
                   'BsmtFinType1', 'BsmtFinType2',
                   'HeatingQC', 'CentralAir', 'Electrical',
                   'KitchenQual', 'Functional','FireplaceQu',
                   'GarageType','GarageFinish',
                   'GarageQual','GarageCond','PavedDrive',
                   'Fence', 'SaleType', 'SaleCondition']

    df[log2_tranf] = df[log2_tranf] + 1
    df[log2_tranf] = np.log2(df[log2_tranf])
    

    for coluna in col_categ_f:
        df[coluna] = df[coluna].astype(str)

    print('Normalização dos dados finalizada.')

    print('Realizando o Feature Select')

    col_select = ['MSSubClass',
                    'MSZoning',
                    'LotFrontage',
                    'LotArea',
                    'Alley',
                    'LotShape',
                    'LandContour',
                    'LotConfig',
                    'LandSlope',
                    'Neighborhood',
                    'Condition1',
                    'BldgType',
                    'HouseStyle',
                    'OverallQual',
                    'OverallCond',
                    'YearBuilt',
                    'YearRemodAdd',
                    'RoofStyle',
                    'Exterior1st',
                    'Exterior2nd',
                    'MasVnrType',
                    'MasVnrArea',
                    'ExterQual',
                    'ExterCond',
                    'Foundation',
                    'BsmtQual',
                    'BsmtCond',
                    'BsmtExposure',
                    'BsmtFinType1',
                    'BsmtFinSF1',
                    'BsmtFinType2',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    'HeatingQC',
                    'CentralAir',
                    'Electrical',
                    'GrLivArea',
                    'BsmtFullBath',
                    'FullBath',
                    'HalfBath',
                    'BedroomAbvGr',
                    'KitchenAbvGr',
                    'KitchenQual',
                    'TotRmsAbvGrd',
                    'Functional',
                    'Fireplaces',
                    'FireplaceQu',
                    'GarageType',
                    'GarageFinish',
                    'GarageCars',
                    'GarageArea',
                    'GarageQual',
                    'GarageCond',
                    'PavedDrive',
                    'WoodDeckSF',
                    'OpenPorchSF',
                    'EnclosedPorch',
                    'Fence',
                    'MoSold',
                    'YrSold',
                    'SaleType',
                    'SaleCondition']
    
    df.drop(df.columns.difference(col_select), axis=1, inplace=True)
    
    print('Feature Select finalizado.')

    print('Pré processamento dos dados finalizado')