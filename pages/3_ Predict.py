import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import Normalizer, OrdinalEncoder
from xgboost import XGBRegressor
import streamlit as st
from art_deploy import *
from category_encoders.target_encoder import TargetEncoder
from PIL import Image

Model = joblib.load('Pickle\Model_Deploy.pkl')
encoder = joblib.load('Pickle/Encoders.pkl')

target_enc = encoder['Target_encod']
normalizer = encoder['Normalizer']

st.header('Previsão')

pref_format = st.selectbox(
    'Como deseja prever o valor?',
    ('Arquivo CSV', 'Inserindo os dados'))

def convert_df(df):
    return df.to_csv().encode('utf-8')

with open('data_description.txt', 'r') as arquivo:
    conteudo = arquivo.read()

if pref_format == 'Arquivo CSV':
    uploaded_file = st.file_uploader("Escolha o arquivo .csv para prever:", type="csv")

    if uploaded_file is not None:

        st.subheader('Dataframe carregado.')
        df = pd.read_csv(uploaded_file)
        st.write("Exibindo os primeiros registros do arquivo:", df.head())
        df_copy = df.copy()

        artefato_deployment(df)
        l1_transf = ['LotArea', 'BsmtFinSF1', 'BsmtUnfSF',  'KitchenAbvGr', 'GarageCars']
        df[l1_transf] = normalizer.transform(df[l1_transf])
        col_categ = df.select_dtypes('object').columns

        df[col_categ] = target_enc.transform(df[col_categ])

        pred = Model.predict(df)
        pred = np.around(pred)


        df_copy['SalePrice'] = pred

        st.subheader('Dataframe com valor previsto.')

        st.write("Exibindo os primeiros registros do arquivo após a predição:", df_copy.head())

        csv_file = convert_df(df_copy)

        st.write('Clique no botão download para baixar o arquivo em CSV com o valor predito.')

        st.download_button(
        label="Download Predict em CSV.",
        data=csv_file,
        file_name='Predict_House.csv',
        mime='text/csv',
    )
elif pref_format == 'Inserindo os dados':
    st.header('Preencha os campos abaixo com os dados da residência.')
    
    st.write('Abaixo Link do dicionário dos dados para auxiliar no preenchimento.')
    st.download_button('Download Dicionário', conteudo)
    
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)
    col9, col10, col11, col12 = st.columns(4)
    col13, col14, col15, col16 = st.columns(4)
    col17, col18, col19, col20 = st.columns(4)
    col21, col22, col23, col24 = st.columns(4)
    col25, col26, col27, col28 = st.columns(4)
    col29, col30, col31, col32 = st.columns(4)
    col33, col34, col35, col36 = st.columns(4)
    col37, col38, col39, col40 = st.columns(4)
    col41, col42, col43, col44 = st.columns(4)
    col45, col46, col47, col48 = st.columns(4)
    col49, col50, col51, col52 = st.columns(4)
    col53, col54, col55, col56 = st.columns(4)
    col57, col58, col59, col60 = st.columns(4)
    col61, col62, col63 = st.columns(3)

    with col1:
        MSSubClass = st.slider('MSSubClass', 0, 190, 0)
        st.write(f"MSSubClass: {MSSubClass}")
        
    with col2:
        MSZoning = st.selectbox('MSZoning',
                                ('RL', 'RM', 'FV','RH', 'C (all)'))
        st.write(f'MSZoning: {MSZoning}')

    with col3:
        LotFrontage = st.slider('LotFrontage', 0, 250, 0)
        st.write(f"LotFrontage: {LotFrontage}")
 
    with col4:
        Alley = st.selectbox('Alley',
                                ('None', 'Grvl', 'Pave'))
        st.write(f'Alley: {Alley}')

    with col5:
        LotShape = st.selectbox('LotShape',
                                ('IR1', 'IR2', 'IR3','Reg'))
        st.write(f'LotShape: {LotShape}')

    with col6:
        LandContour = st.selectbox('LandContour',
                                ('Low', 'Lv1', 'HLS','bNK','Low'))
        st.write(f'LandContour: {LandContour}')

    with col7:
        LotConfig = st.selectbox('LotConfig',
                                ('Inside', 'Corner', 'CulDSac','FR2','FR3'))
        st.write(f'LotConfig: {LotConfig}')

    with col8:
        LandSlope = st.selectbox('LandSlope',
                                ('Gtl', 'Mod', 'Sev'))
        st.write(f'LandSlope: {LandSlope}')

    with col9:
        Neighborhood = st.selectbox('Neighborhood',
                                ('Blueste', 'Blmngtn', 'Veenker','NPkVill','BrDale','BrDale','ClearCr','MeadowV','SWISU','StoneBr', 'NoRidge', 'Timber', 'BrkSide','Crawfor'
                                'IDOTRR','NWAmes', 'Mitchel', 'Mitchel','SawyerW','Sawyer','Gilbert','NridgHt', 'Edwards', 'Somerst', 'CollgCr', 'CollgCr', 'NAmes','OldTown'))
        st.write(f'Neighborhood: {Neighborhood}')

    with col10:
        Condition1 = st.selectbox('Condition1',
                                ('Norm', 'Feedr', 'Artery','RRAn','PosN','RRAe','PosA','RRNe','RRNn'))
        st.write(f'Condition1: {Condition1}')

    with col11:
        BldgType = st.selectbox('BldgType',
                                ('1Fam', 'TwnhsE', 'Duplex','Twnhs','2fmCon'))
        st.write(f'BldgType: {BldgType}')

    with col12:
        HouseStyle = st.selectbox('HouseStyle',
                                ('1Story', '2Story', '1.5Fin', 'SLvl', 'SFoyer', '2.5Unf', '1.5Unf'))
        st.write(f'HouseStyle: {HouseStyle}')

    with col13:
        OverallQual = st.slider('OverallQual', 1, 10, 5)
        st.write(f"OverallQual: {OverallQual}")

    with col14:
        OverallCond = st.slider('OverallCond', 1, 10, 5)
        st.write(f"OverallQual: {OverallCond}")

    with col15:
        YearBuilt = st.slider('YearBuilt', 1800, 2023, 2000)
        st.write(f"YearBuilt: {YearBuilt}")

    with col16:
        YearRemodAdd = st.slider('YearRemodAdd', 1800, 2023, 2000)
        st.write(f"YearRemodAdd: {YearRemodAdd}")

    with col17:
        RoofStyle = st.selectbox('RoofStyle',
                                ('Gable', 'Hip', 'Gambrel', 'Flat', 'Mansard', 'Shed'))
        st.write(f"RoofStyle: {RoofStyle}")

    with col18:
        Exterior1st = st.selectbox('Exterior1st',
                                ('VinylSd', 'MetalSd', 'HdBoard', 'Wd Sdng', 'Plywood', 'CemntBd', 'BrkFace', 'WdShing', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'CBlock','None'))
        st.write(f"Exterior1st: {Exterior1st}")

    with col19:
        Exterior2nd = st.selectbox('Exterior2nd',
                                ('VinylSd', 'MetalSd', 'HdBoard', 'Wd Sdng', 'Plywood', 'CemntBd', 'BrkFace', 'Wd Shng', 'AsbShng', 'Stucco', 'Brk Cmn', 'AsphShn', 'CBlock', 'Stone', 'ImStucc','None'))
        st.write(f"Exterior2nd: {Exterior2nd}")

    with col20:
        MasVnrType = st.selectbox('MasVnrType',
                                ('None', 'BrkFace', 'Stone', 'BrkCmn','None'))
        st.write(f"MasVnrType: {MasVnrType}")

    with col21:
        MasVnrArea = st.slider('MasVnrArea', 0, 1500, 100)
        st.write(f"MasVnrArea: {MasVnrArea}")

    with col22:
        ExterCond = st.selectbox('ExterCond',
                                ('TA', 'Gd', 'Ex', 'Fa'))
        st.write(f"ExterCond: {ExterCond}")

    with col23:
        ExterQual = st.selectbox('ExterQual',
                                ('TA', 'Gd', 'Ex', 'Fa','Po'))
        st.write(f"ExterQual: {ExterQual}")

    with col24:
        Foundation = st.selectbox('Foundation',
                                ('PConc', 'CBlock', 'BrkTil', 'Slab','Stone','Wood'))
        st.write(f"Foundation: {Foundation}")
             
    with col25:
        BsmtQual = st.selectbox('BsmtQual',
                                ('TA', 'Gd', 'Ex', 'Fa','None'))
        st.write(f"BsmtQual: {BsmtQual}")

    with col26:
        BsmtCond = st.selectbox('BsmtCond',
                                ('TA', 'Gd', 'Po', 'Fa','None'))
        st.write(f"BsmtCond: {BsmtCond}")

    with col27:
        BsmtExposure = st.selectbox('BsmtExposure',
                                ('No', 'Av', 'Gd', 'Mn','None'))
        st.write(f"BsmtExposure: {BsmtExposure}")

    with col28:
        BsmtFinType1 = st.selectbox('BsmtFinType1',
                                ('GLQ', 'Unf', 'ALQ', 'Rec','BLQ','LwQ', 'None'))
        st.write(f"BsmtFinType1: {BsmtFinType1}")

    with col29:
        BsmtFinSF1 = st.slider('BsmtFinSF1', 0, 4200, 100)
        st.write(f"BsmtFinSF1: {BsmtFinSF1}")

    with col30:
        BsmtFinType2 = st.selectbox('BsmtFinType2',
                                ('Unf', 'Rec', 'None', 'LwQ','BLQ','ALQ', 'GLQ'))
        st.write(f"BsmtFinType2: {BsmtFinType2}")

    with col31:
        BsmtFinSF2 = st.slider('BsmtFinSF2', 0, 1800, 100)
        st.write(f"BsmtFinSF2: {BsmtFinSF2}")

    with col32:
        BsmtUnfSF = st.slider('BsmtUnfSF', 0, 2500, 100)
        st.write(f"BsmtUnfSF: {BsmtUnfSF}")

    with col33:
        TotalBsmtSF = st.slider('TotalBsmtSF', 0, 6000, 1000)
        st.write(f"TotalBsmtSF: {TotalBsmtSF}")

    with col34:
        HeatingQC = st.selectbox('HeatingQC',
                                ('Ex', 'TA', 'Gd', 'Fa','Po'))
        st.write(f"HeatingQC: {HeatingQC}")

    with col35:
        CentralAir = st.selectbox('CentralAir',
                                ('Y', 'N'))
        st.write(f"CentralAir: {CentralAir}")

    with col36:
        Electrical = st.selectbox('Electrical',
                                ('SBrkr', 'FuseA','FuseF','FuseP'))
        st.write(f"Electrical: {Electrical}")

    with col37:
        GrLivArea = st.slider('GrLivArea', 5, 15, 10)
        st.write(f"GrLivArea: {GrLivArea}")

    with col38:
        BsmtFullBath = st.slider('BsmtFullBath', 0, 5, 1)
        st.write(f"BsmtFullBath: {BsmtFullBath}")
    
    with col39:
        FullBath = st.slider('FullBath', 0, 5, 1)
        st.write(f"FullBath: {FullBath}")

    with col40:
        HalfBath = st.slider('HalfBath', 0, 3, 1)
        st.write(f"HalfBath: {HalfBath}")
    
    with col41:
        BedroomAbvGr = st.slider('BedroomAbvGr', 0, 8, 3)
        st.write(f"HalfBath: {BedroomAbvGr}")
    
    with col42:
        KitchenQual = st.selectbox('KitchenQual',
                                ('TA', 'Gd','Ex','Fa','0'))
        st.write(f"KitchenQual: {KitchenQual}")
    
    with col43:
        TotRmsAbvGrd = st.slider('TotRmsAbvGrd', 1, 5, 3)
        st.write(f"TotRmsAbvGrd: {TotRmsAbvGrd}")
    
    with col44:
        Functional = st.selectbox('Functional',
                                ('Typ', 'Min2','Min1','Mod','0','Maj1','Maj2','Sev'))
        st.write(f"Functional: {Functional}")

    with col45:
        Fireplaces = st.slider('Fireplaces', 0, 3, 1)
        st.write(f"Fireplaces: {Fireplaces}")
    
    with col46:
        FireplaceQu = st.selectbox('FireplaceQu',
                                ('None', 'Gd','TA','Fa','Po','Ex'))
        st.write(f"FireplaceQu: {FireplaceQu}")
    
    with col47:
        GarageType = st.selectbox('GarageType',
                                ('Attchd', 'Detchd','BuiltIn','None','Basment','2Types','CarPort'))
        st.write(f"GarageType: {GarageType}")

    with col48:
        GarageFinish = st.selectbox('GarageFinish', 
                                    ('Unf', 'RFn', 'Fin', 'None'))
        st.write(f"GarageFinish: {GarageFinish}")

    with col49:
        GarageArea = st.slider('GarageArea', 0, 1700, 480)
        st.write(f"GarageArea: {GarageArea}")
    
    with col50:
        GarageQual = st.selectbox('GarageQual', 
                                    ('TA', 'None', 'Fa', 'Gd', 'Po'))
        st.write(f"GarageQual: {GarageQual}")
    
    with col51:
        GarageCond = st.selectbox('GarageCond', 
                                    ('TA', 'None', 'Fa', 'Gd', 'Po', 'Ex'))
        st.write(f"GarageCond: {GarageCond}")
    
    with col52:
        PavedDrive = st.selectbox('PavedDrive', 
                                    ('Y', 'N', 'P'))
        st.write(f"PavedDrive: {PavedDrive}")

    with col53:
        WoodDeckSF = st.slider('WoodDeckSF', 0, 11, 5)
        st.write(f"WoodDeckSF: {WoodDeckSF}")

    with col54:
        OpenPorchSF = st.slider('OpenPorchSF', 0, 11, 5)
        st.write(f"OpenPorchSF: {OpenPorchSF}")
    
    with col55:
        EnclosedPorch = st.slider('EnclosedPorch', 0, 10, 0)
        st.write(f"EnclosedPorch: {EnclosedPorch}")
    
    with col56:
        Fence = st.selectbox('Fence', 
                                    ('None', 'MnPrv', 'GdPrv', 'GdWo', 'MnWw'))
        st.write(f"Fence: {Fence}")
    
    with col57:
        MoSold = st.slider('MoSold', 1, 12, 6)
        st.write(f"MoSold: {MoSold}")

    with col58:
        SaleType = st.selectbox('SaleType', 
                                    ('0', 'WD', 'New', 'COD', 'ConLD', 'CWD','Oth','ConLI','Con','ConLW'))
        st.write(f"SaleType: {SaleType}")

    with col59:
        SaleCondition = st.selectbox('SaleCondition', 
                                    ('Normal', 'Partial', 'Abnormal', 'Family', 'Alloca', 'AdjLand'))
        st.write(f"SaleCondition: {SaleCondition}")
    
    with col60:
        GarageCars = st.slider('GarageCars', 0, 6, 2)
        st.write(f"GarageCars: {GarageCars}")
    
    with col61:
        KitchenAbvGr = st.slider('KitchenAbvGr', 0, 3, 1)
        st.write(f"KitchenAbvGr: {KitchenAbvGr}")
    
    with col62:
        LotArea = st.slider('LotArea', 500, 60000, 9000)
        st.write(f"LotArea: {LotArea}")
    
    with col63:
        YrSold = st.slider('YrSold', 2006, 2023, 2008)
        st.write(f"YrSold: {YrSold}")



    if st.button('Prever valor'):
        
        v = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Alley', 'LotShape',
                 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
                 'YearRemodAdd', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
                 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC',
                 'CentralAir', 'Electrical', 'GrLivArea', 'BsmtFullBath', 'FullBath',
                 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
                 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
                 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
                 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'Fence',
                 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
        data_prev = dict(zip(['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Alley', 'LotShape',
                 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
                 'YearRemodAdd', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
                 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC',
                 'CentralAir', 'Electrical', 'GrLivArea', 'BsmtFullBath', 'FullBath',
                 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
                 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
                 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
                 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'Fence',
                 'MoSold', 'YrSold', 'SaleType', 'SaleCondition'],
                [MSSubClass, MSZoning, LotFrontage, LotArea, Alley, LotShape,
                 LandContour, LotConfig, LandSlope, Neighborhood, Condition1,
                 BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt,
                 YearRemodAdd, RoofStyle, Exterior1st, Exterior2nd, MasVnrType,
                 MasVnrArea, ExterQual, ExterCond, Foundation, BsmtQual,
                 BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinSF1,
                 BsmtFinType2, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, HeatingQC,
                 CentralAir, Electrical, GrLivArea, BsmtFullBath, FullBath,
                 HalfBath, BedroomAbvGr, KitchenAbvGr, KitchenQual,
                 TotRmsAbvGrd, Functional, Fireplaces, FireplaceQu, GarageType,
                 GarageFinish, GarageCars, GarageArea, GarageQual, GarageCond,
                 PavedDrive, WoodDeckSF, OpenPorchSF, EnclosedPorch, Fence,
                 MoSold, YrSold, SaleType, SaleCondition
                 ]))
        
        df_prev = pd.DataFrame(data_prev, index=[0], columns=v)
        st.write('Dados inseridos no formato de dataframe: ',df_prev.head(1))

        # Criando uma cópia do dataset gerado onde iremos aplicar as transformações e fazer o predict.
        df_pred = df_prev.copy()
        l1_transf = ['LotArea', 'BsmtFinSF1', 'BsmtUnfSF',  'KitchenAbvGr', 'GarageCars']
        log2_tranf = ['MSSubClass', 'OverallCond', 'MasVnrArea', 'BsmtFinSF2', 'GrLivArea', 'BsmtFullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']
        col_categ = df_pred.select_dtypes('object').columns

        df_pred[log2_tranf] = df_pred[log2_tranf] + 1
        df_pred[log2_tranf] = np.log2(df_pred[log2_tranf])
        df_pred[l1_transf] = normalizer.transform(df_pred[l1_transf])
        df_pred[col_categ] = target_enc.transform(df_pred[col_categ])

        prev = Model.predict(df_pred)
        df_prev['SalePrice'] = prev
        image = Image.open('Imagens\house.jpg')
        st.image(image, caption='Imagem meramente ilustrativa')

        st.markdown(f'**A residência possui um valor estimado de: {prev}**')

        st.write('Clique no botão abaixo para baixar dados e o valor da casa no formato .CSV')

        csv_file_pred = convert_df(df_prev)
        st.download_button(
        label="Download Predict em CSV.",
        data=csv_file_pred,
        file_name='House_Value.csv',
        mime='text/csv',)