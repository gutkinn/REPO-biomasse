from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import xarray as xr
import os
from shapely import geometry, buffer
import geopandas as gpd
import json
from scipy import stats

def select_model(model):
    if model == 'RF' or model == 'PLS':
        mode = model
        degree = None
    elif model == 'LR':
        mode = model
        degree = 1
    elif model == 'PR2':
        mode='LR'
        degree = 2
    elif model == 'PR3':
        mode='LR'
        degree = 3
    else:
        raise Exception('Erreur : modèle incorrect')
    
    return mode, degree

def create_gpkg(in_csv):
    sites_csv = pd.read_csv(in_csv)
    sites_points = [geometry.Point(x,y) for x,y in zip(sites_csv['Long (E)'],sites_csv['Lat (N)'])]
    sites_gdf = gpd.GeoDataFrame(sites_csv,crs='epsg:4326',geometry=sites_points)
    sites_gdf['geometry'] = buffer(sites_gdf.geometry.to_crs(3857),30).set_crs(3857).to_crs(4326)
    out_gpkg = in_csv.replace('.csv','.gpkg')
    sites_gdf.to_file(out_gpkg)

    prodsites = gpd.read_file(out_gpkg)

    return prodsites

def generate_geometries(prodsites):

    # bandes satellitaires Sentinel-1 et Sentinel-2 dont nous aurons besoin
    band_list_s1 = ['VV','VH']
    band_list_s2 = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']

    # ici nous réorganisons les données de biomasse et les dates de collection
    all_dates = ['date_fev','date_avr','date_sep']
    all_biomass = ['biomass_fev','biomass_avr','biomass_sep']
    prodsites['dates'] = prodsites.apply(create_list,axis=1,args=(all_dates,))
    prodsites['biomass'] = prodsites.apply(create_list,axis=1,args=(all_biomass,True))

    # création des géométries pour l'extraction des données satellitaires 
    bbox_simple = {"west":prodsites.geometry.to_crs(crs=3857).centroid.to_crs(4326).x.min()-0.0001,
                   "south":prodsites.geometry.to_crs(crs=3857).centroid.to_crs(4326).y.min()-0.0001,
                   "east":prodsites.geometry.to_crs(crs=3857).centroid.to_crs(4326).x.max()+0.0001,
                   "north":prodsites.geometry.to_crs(crs=3857).centroid.to_crs(4326).y.max()+0.0001}

    geoms = json.loads(prodsites.explode(index_parts=True).geometry.to_json())

    return geoms, bbox_simple, prodsites, band_list_s1, band_list_s2

def generate_geometries_zone(loc,t_extract):

    mois = t_extract[0].split('-')[1]
    if loc == 'test':
        area = gpd.read_file(f'./in_data/test_zone.gpkg').to_crs(epsg=4326)
    else:
        area = gpd.read_file(f'./in_data/communes/{loc}.gpkg').to_crs(epsg=4326)
    bounds = area.total_bounds
    in_data_path = os.path.join('.','out_data',f'{loc}_{mois}_100m.nc')
    
    bbox = {'west':bounds[0],
            'south':bounds[1],
            'east':bounds[2],
            'north':bounds[3],
            'crs':area.crs.to_string()}

    return bbox, area, mois, in_data_path

def get_in_data(in_data_path):
    in_data = xr.open_dataset(in_data_path)
    add_indices(in_data,rolling=True)
    in_crs = in_data['crs'].attrs['crs_wkt']
    in_data = in_data.drop_vars(['crs']).mean(dim='t')
    return in_data, in_crs

def generate_predictions(model_type, X_df, y_df, in_data, in_crs, out_path, area):
    mode, degree = select_model(model_type)

    if mode == 'RF':
        mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode=mode)
        model = get_best_model(X_df[remaining_feats[max(r2_scores,key=r2_scores.get)]],y_df,mode)
        predict_and_save(model,in_data,in_crs,out_path,area.geometry,it_mode = mode)
    elif mode == 'PLS':
        mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode=mode)
        model = get_best_model(X_df[remaining_feats[21]],y_df,mode,degree)
        coef = dict(zip(remaining_feats[21],model.coef_[0]))
        predict_and_save(model,in_data,in_crs,out_path,area.geometry,it_mode=mode,coef=coef)
    else:
        mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode=mode,degree=degree)
        model = get_best_model(X_df[remaining_feats[max(r2_scores,key=r2_scores.get)]],y_df,mode,degree)
        coef = dict(zip(remaining_feats[max(r2_scores,key=r2_scores.get)],model.coef_[0]))
        predict_and_save(model,in_data,in_crs,out_path,area.geometry,it_mode=mode,coef=coef,degree=degree)        

def preprocess_data(prodsites,extr_data,all_indexes,band_list_s1,band_list_s2):
    # ici nous allons agréger les données biomasse au niveau du pixel (2 groupes dans chaque pixel)
    list_l1 = list(np.unique([p.split('_')[0] + '_' + p.split('_')[1] for p in prodsites['ID Point'] if '45_' not in p]))

    biomass_list = []
    i=0

    for idx_l1 in list_l1:
        
        merge_biomass, merge_rows = match_satellite_biomass(idx_l1,extr_data,prodsites,band_list_s1 + band_list_s2)
        biomass_vals, matched_index_vals = generate_corr(merge_biomass,merge_rows,all_indexes,0)
        biomass_list+=biomass_vals

        if i==0:
            index_list = np.array(matched_index_vals)
        else:
            index_list = np.append(index_list,np.array(matched_index_vals),axis=1)
        i+=1
            
    # réorganisation des données temporelles (t1 = février, t2 = avril, t3 = septembre)
    all_l1_time = np.array([[l+'_t1',l+'_t2',l+'_t3'] for l in list_l1]).reshape(index_list.shape[1])

    # nettoyage des données biomasse, données avec score z > 2 sont retirées
    z_b = np.abs(stats.zscore(biomass_list))
    z_b_o = np.where(z_b > 2)[0]
    index_list = np.array(pd.DataFrame(index_list).drop(columns=z_b_o))
    biomass_list = np.delete(biomass_list,z_b_o)
    all_l1_time = np.delete(all_l1_time,z_b_o)

    return index_list, biomass_list, all_l1_time

def open_dataset(prodsites,points_out):
    dict_dates = {}

    extr_data = pd.concat([prodsites['ID Point'],pd.read_json(points_out,convert_axes=False)],axis=1)
    extr_data = extr_data[['ID Point']+list(extr_data.keys()[1:].sort_values())]

    for key in extr_data.columns.values[1:]:
        dict_dates[key] = key[:10]
    extr_data.rename(columns=dict_dates,inplace=True)

    return extr_data

def create_id_point(row):
    return f"{row['Site']}_{row['Pixel']}_{row['Groupe']}"

def create_list(row,col_names,float_flag=False):
    if float_flag:
        return [float(row[col_names[0]]),float(row[col_names[1]]),float(row[col_names[2]])]
    else:
        return [row[col_names[0]],row[col_names[1]],row[col_names[2]]]

def extract_and_save_simple(b1,b2,t,bbox,geom,conn):
    s1_cube = conn.load_collection(
        "SENTINEL1_GRD",
        spatial_extent=bbox,
        temporal_extent = t,
        bands = b1
    )
    s1_cube = s1_cube.sar_backscatter(coefficient = 'gamma0-ellipsoid')

    s2_cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent = t,
        bands = b2
    )

    scl = conn.load_collection(
        "SENTINEL2_L2A",
        bands=["SCL"],
        spatial_extent=bbox,
        temporal_extent = t,
        max_cloud_cover=85
    )

    mask = scl.process("to_scl_dilation_mask",data=scl)
    s2_masked = s2_cube.mask(mask)
    
    merged_cubes = s1_cube.merge_cubes(s2_masked)
    merged_cubes = merged_cubes.apply_dimension(dimension="t", process="array_interpolate_linear")
    
    if geom != None:
        agg_cubes = merged_cubes.aggregate_spatial(geom, reducer="mean")
    else: 
        agg_cubes = merged_cubes
        agg_cubes = agg_cubes.resample_spatial(resolution=100,projection='epsg:3857')
        agg_cubes = agg_cubes.resample_spatial(resolution=0,projection='epsg:4326')
    return agg_cubes

def norm_diff(arr1, arr2):
    return (arr1 - arr2) / (arr1 + arr2)

def add_indices(row_data,rolling=True):
    
    vv = row_data['VV']
    vh = row_data['VH']
    b02 = row_data['B02']
    b03 = row_data['B03']
    b04 = row_data['B04']
    b05 = row_data['B05']
    b06 = row_data['B06']
    b07 = row_data['B07']
    b08 = row_data['B08']
    b11 = row_data['B11']
    b12 = row_data['B12']

    row_data['VV-VH'] = vv - vh
    row_data['VV+VH'] = vv + vh
    
    row_data['NDVI'] = norm_diff(b08,b04)
    row_data['NDWI'] = norm_diff(b08,b11)
    row_data['GRVI'] = norm_diff(b03,b04)
    row_data['GNDVI'] = norm_diff(b08,b03)
    row_data['NBR'] = norm_diff(b08,b12)
    row_data['NBR2'] = norm_diff(b11,b12)
    row_data['NDI54'] = norm_diff(b05,b04)
    row_data['NDRE1'] = norm_diff(b08,b05)
    
    row_data['STI'] = b11 / b12
    row_data['MSI'] = b11 / b08

    row_data['SAVI'] = ((b08 - b04) / (b08 + b04 + 0.5)) * 1.5
    row_data['MSAVI'] = ((2 * b08) + 1 - np.sqrt((2 * b08)**2 - 8*(b08 - b04))) / 2
    row_data['IRECI'] = b07 - (b04/(b05/b06))

    if rolling:
        if type(row_data['VV-VH']) == xr.DataArray:
            row_data['VV-VH_rolling'] = row_data['VV-VH'].to_dataframe().rolling(3,min_periods=1).mean().to_xarray()['VV-VH']
            row_data['VV+VH_rolling'] = row_data['VV+VH'].to_dataframe().rolling(3,min_periods=1).mean().to_xarray()['VV+VH']
        else:
            row_data['VV-VH_rolling'] = row_data['VV-VH'].rolling(3,min_periods=1).mean()
            row_data['VV+VH_rolling'] = row_data['VV+VH'].rolling(3,min_periods=1).mean()

    return row_data

def arrange_sat_data(subsets,band_list):
    rows = []
    for subset in subsets:
        rows.append(add_indices(pd.concat([pd.Series(subset.columns.values[1:],name='date'),
                        pd.DataFrame(list(subset.values[0][1:]),columns=band_list)],axis=1).set_index('date')))
    return rows

def match_dates(biomass_dates,index_dates):
    nearest_dates = []
    for date in biomass_dates:
        nearest = min(index_dates,key=lambda x: abs(x - date))
        nearest_dates.append(nearest)
    return nearest_dates

def generate_corr(biomass_row,index_df,index_types,adj_factor=0):
    """
    Generates a set of index values matched to biomass values
    Adj factor shifts number of days to match
    """
    biomass_dates = np.array(biomass_row.dates).astype('datetime64[s]')
    biomass_dates = np.array([date + np.timedelta64(adj_factor,'D') for date in biomass_dates])
    biomass_vals = [float(val) for val in biomass_row.biomass]
    matched_index_vals = []
    for index in index_types:
        index_row = index_df[index]
        index_row.index = index_row.index.astype('datetime64[s]')
        index_dates = index_row.index.values

        matched_index_dates = match_dates(biomass_dates,index_dates)
        matched_index_vals.append([index_row[date] for date in matched_index_dates])
    return biomass_vals,matched_index_vals

def get_fs(X_train,y_train,X_test,k):
    # configure to select all features
    fs = SelectKBest(score_func = mutual_info_regression, k=k)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def iterate_train_test(X_df,y_df,split=0.25,it_mode='RF',degree=2):
    mae_scores={}
    r2_scores = {}
    remaining_feats = {}
    preds = {}
    
    for run in range(len(X_df.columns)):
        # split into train and test sets
        if split > 0:
            X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_df,y_df,test_size=split, random_state=1)
            X_train = X_train_p.values
            X_test = X_test_p.values
            y_train = y_train_p.values.ravel()
            y_test = y_test_p.values.ravel()
        else:
            # if training using whole dataset
            X_train = X_df.values
            X_test = X_df.values
            y_train = y_df.values.ravel()
            y_test = y_df.values.ravel()

        if it_mode == 'RF':
            remaining_feats[run] = X_df.columns.values
            regressor = RandomForestRegressor(n_estimators=50,random_state=0, oob_score=True)

            regressor.fit(X_train,y_train)
            # Making predictions on the same data or new data
            predictions = regressor.predict(X_test)
            
            # Evaluating the model
            mae = mean_absolute_error(y_test, predictions)
            mae_scores[run] = mae

            r2 = r2_score(y_test, predictions)
            r2_scores[run] = r2

            feat_imp = {feat:imp for feat,imp in zip(X_df.columns, regressor.feature_importances_)}
            drop_val = pd.DataFrame(feat_imp,index=['importance']).T.sort_values(by='importance')[:1].index
            #drop least important values
            X_df = X_df.drop(columns=drop_val)
            preds[run] = predictions

        elif it_mode == 'LR':
            
            k = len(X_df.columns) - run
            X_train_fs, X_test_fs, fs = get_fs(X_train,y_train,X_test,k)

            poly = PolynomialFeatures(degree, include_bias=False)
            poly_train = poly.fit_transform(X_train_fs)
            poly_test = poly.fit_transform(X_test_fs)
            model = LinearRegression()
            model.fit(poly_train,y_train)
            y_pred = model.predict(poly_test)
            mae = mean_absolute_error(y_test,y_pred)
            r2 = model.score(poly_test,y_test)
            mae_scores[run] = mae
            r2_scores[run] = r2

            cols_idxs = fs.get_support(indices=True)
            remaining_feats[run] = X_df.columns.values[cols_idxs]
            preds[run] = y_pred
            continue

        elif it_mode == 'PLS':

            k = len(X_df.columns)

            model = PLSRegression(n_components=k)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test,y_pred)
            r2 = model.score(X_test,y_test)

            mae_scores[run] = mae
            r2_scores[run] = r2

            drop_val = X_df.columns[np.argsort(np.abs(model.coef_[0]))[0]]

            X_df = X_df.drop(columns=drop_val)
            remaining_feats[run] = X_df.columns.values
            preds[run] = y_pred
            continue


    return mae_scores,r2_scores,remaining_feats,[y_test,preds]

def match_satellite_biomass(idx_l1,extr_data,prodsites,band_list):
    
    matching_l1 = [r for r in prodsites['ID Point'] if r.startswith(idx_l1)]
                
    row_0 = extr_data[extr_data['ID Point']==matching_l1[0]]
    row_0 = arrange_sat_data([row_0],band_list)[0].dropna()
    row_1 = extr_data[extr_data['ID Point']==matching_l1[1]]
    row_1 = arrange_sat_data([row_1],band_list)[0].dropna()
    merge_rows = pd.concat([row_0,row_1]).groupby(level=0).mean()
    brow_0 = prodsites[prodsites['ID Point']==matching_l1[0]]['biomass'].values[0]
    brow_1 = prodsites[prodsites['ID Point']==matching_l1[1]]['biomass'].values[0]

    merge_biomass = pd.Series({'ID Point':idx_l1,
                               'dates':prodsites[prodsites['ID Point']==matching_l1[0]]['dates'].values[0],
                               'biomass':[adjust_biomass(np.mean([g,h])) for g,h in zip(brow_0,brow_1)]})
    
    return merge_biomass, merge_rows

def get_best_model(X_df,y_df,it_mode,degree=1):
    if it_mode == 'RF':
        regressor = RandomForestRegressor(n_estimators=50,random_state=0, oob_score=True)
        regressor.fit(X_df,y_df)
        return regressor
    
    if it_mode == 'PLS':
        model = PLSRegression(n_components=len(X_df.columns))
        model.fit(X_df,y_df)
        return model
    
    elif it_mode == 'LR':
        poly = PolynomialFeatures(degree, include_bias=False)
        poly_train = poly.fit_transform(X_df)
        model = LinearRegression()
        model.fit(poly_train,y_df)
        
        if degree == 1:
            return model
        
        elif degree >=2:
            return model, poly

def get_coef_calcs(in_data,coef,degree):
    if degree <= 1:
        return [(coef[key] * in_data[key]) for key in coef]
    if degree >= 2:
        coef_calcs = []
        for key in coef:
            if ' ' in key:
                if '^2' in key:
                    band_1,band_2 = key.split(' ')
                    if '^2' in band_1:
                        band_1 = band_1.split('^')[0]
                        coef_calcs.append(coef[key] * (in_data[band_1] * in_data[band_1] * in_data[band_2]))
                    elif '^2' in band_2:
                        band_2 = band_2.split('^')[0]
                        coef_calcs.append(coef[key] * (in_data[band_1] * in_data[band_2] * in_data[band_2]))
                else:
                    band_1,band_2 = key.split(' ')
                    coef_calcs.append(coef[key] * (in_data[band_1] * in_data[band_2]))

            elif '^2' in key:
                band_name = key.split('^')[0]
                coef_calcs.append(coef[key] * (in_data[band_name] * in_data[band_name]))
            elif '^3' in key:
                band_name = key.split('^')[0]
                coef_calcs.append(coef[key] * (in_data[band_name] * in_data[band_name] * in_data[band_name]))
            else:
                coef_calcs.append(coef[key] * in_data[key])
    return coef_calcs

def adjust_biomass(in_data):
    in_data = ((in_data * 30) / 0.5) / 3
    return in_data

def generate_formula(coef,intercept):
    coef_str = ''
    for c in coef:
        coef_str += f'({coef[c]} * {c})'
        if c != list(coef.keys())[-1]:
            coef_str += ' + '
    formula = str(intercept) + ' + ' + coef_str
    return formula

def predict_and_save(model,in_data,crs,out_path,geom,it_mode,coef=None,degree=0):
    if it_mode == 'PLS':
        pred = xr.DataArray(data = model.predict(
                in_data[model.feature_names_in_].to_dataframe().reset_index(drop=True).fillna(0)).reshape(
                                                                                        len(in_data[model.feature_names_in_].y),
                                                                                        len(in_data[model.feature_names_in_].x)),
            dims = ['y','x'], 
            coords = dict(y=('y',in_data[model.feature_names_in_]['y'].data),
                                  x=('x',in_data[model.feature_names_in_]['x'].data)))
        pred = pred.to_dataset(name='biomasse')
        pred = pred.where(pred['biomasse'] > 0, 0)
        pred.rio.write_crs(crs, inplace=True)
        pred = pred.rio.clip(geom,crs,drop=True)
        pred['biomasse'].plot()
        pred.rio.to_raster(out_path.replace('MODEL','PLS'))     

    if it_mode == 'LR':
        coef_calcs = get_coef_calcs(in_data,coef,degree)
        intercept = model.intercept_[0]

        formula = generate_formula(coef,intercept)
        with open(os.path.join('.','out_data',f'formule_{it_mode}{degree}.txt'),'w') as f:
            f.write('Formule régression linéaire:')
            f.write('\n')
            f.write(formula)
        f.close()

        pred = intercept + sum(coef_calcs)
        pred = pred.to_dataset(name='biomasse')
        pred = pred.where(pred['biomasse'] > 0, 0)
        pred.rio.write_crs(crs, inplace=True)
        pred = pred.rio.clip(geom,crs,drop=True)
        pred['biomasse'].plot()
        pred.rio.to_raster(out_path.replace('MODEL',f'LR{degree}'))
    
    if it_mode == 'RF':
        img = np.array(in_data[model.feature_names_in_].to_dataarray())
        new_img = img.reshape(img.shape[0],(img.shape[1] * img.shape[2]))
        new_img[np.isnan(new_img)] = 0
        RF_results = model.predict(pd.DataFrame(new_img.T,columns = model.feature_names_in_ ))
        pred = xr.DataArray(data = RF_results.reshape(img.shape[1],img.shape[2]),
                    dims = ['y','x'],
                    coords = dict(y=('y',in_data['y'].data),
                                  x=('x',in_data['x'].data)))
        pred = pred.to_dataset(name='biomasse')
        pred = pred.where(pred['biomasse']>0,0)
        pred.rio.write_crs(crs, inplace=True)
        pred = pred.rio.clip(geom,crs,drop=True)
        pred['biomasse'].plot()
        pred.rio.to_raster(out_path.replace('MODEL','RF'))

def print_models(X_df,y_df):
    print('Modèle régression linéaire (1°)')
    mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode='LR',degree=1)
    print(f'Min MAE: itération {min(mae_scores,key=mae_scores.get)} score {np.round(mae_scores[min(mae_scores,key=mae_scores.get)],3)}')
    print(f'Max R²: itération {max(r2_scores,key=r2_scores.get)} score {np.round(r2_scores[max(r2_scores,key=r2_scores.get)],3)}')
    print(f'Bandes restantes : {remaining_feats[max(r2_scores,key=r2_scores.get)]}')
    m_LR = max(r2_scores,key=r2_scores.get)
    print('\n')

    print('Modèle régression polynomiale (2°)')
    mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode='LR',degree=2)
    print(f'Min MAE: itération {min(mae_scores,key=mae_scores.get)} score {np.round(mae_scores[min(mae_scores,key=mae_scores.get)],3)}')
    print(f'Max R²: itération {max(r2_scores,key=r2_scores.get)} score {np.round(r2_scores[max(r2_scores,key=r2_scores.get)],3)}')
    print(f'Bandes restantes : {remaining_feats[max(r2_scores,key=r2_scores.get)]}')
    m_PR2 = max(r2_scores,key=r2_scores.get)
    print('\n')

    print('Modèle régression polynomiale (3°)')
    mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode='LR',degree=3)
    print(f'Min MAE: itération {min(mae_scores,key=mae_scores.get)} score {np.round(mae_scores[min(mae_scores,key=mae_scores.get)],3)}')
    print(f'Max R²: itération {max(r2_scores,key=r2_scores.get)} score {np.round(r2_scores[max(r2_scores,key=r2_scores.get)],3)}')
    print(f'Bandes restantes : {remaining_feats[max(r2_scores,key=r2_scores.get)]}')
    m_PR3 = max(r2_scores,key=r2_scores.get)
    print('\n')

    print('Modèle moindre carrés partiels (PLS)')
    mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode='PLS')
    print(f'Min MAE : itération {min(mae_scores,key=mae_scores.get)} score {np.round(mae_scores[min(mae_scores,key=mae_scores.get)],3)}')
    print(f'Max R² : itération {max(r2_scores,key=r2_scores.get)} score {np.round(r2_scores[max(r2_scores,key=r2_scores.get)],3)}')
    print(f'Bandes restantes : {remaining_feats[max(r2_scores,key=r2_scores.get)]}')
    m_PLS = max(r2_scores,key=r2_scores.get)
    print('\n')

    print('Modèle Random Forests (RF)')
    mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode='RF')
    print(f'Min MAE : itération {min(mae_scores,key=mae_scores.get)} score {np.round(mae_scores[min(mae_scores,key=mae_scores.get)],3)}')
    print(f'Max R² : itération {max(r2_scores,key=r2_scores.get)} score {np.round(r2_scores[max(r2_scores,key=r2_scores.get)],3)}')
    print(f'Bandes restantes : {remaining_feats[max(r2_scores,key=r2_scores.get)]}')
    m_RF = max(r2_scores,key=r2_scores.get)

    return {'LR':m_LR,'PR2':m_PR2,'PR3':m_PR3,'PLS':m_PLS,'RF':m_RF}

def draw_plot(model,X_df,y_df):

    mode,degree = select_model(model)

    mae_scores,r2_scores,remaining_feats,[y_test,preds] = iterate_train_test(X_df,y_df,split=0.25,it_mode=mode,degree=degree)
    f,ax = plt.subplots(figsize=(6,5))
    plt.title(f'Modèle {mode} degré {degree}')

    ax.plot([n for n in range(len(r2_scores))],r2_scores.values(),color='steelblue')
    ax2 = ax.twinx()
    ax2.plot([n for n in range(len(r2_scores))],mae_scores.values(),color='darkred')
    ax.set_ylabel('R² score (bleu)')
    ax.set_yscale('symlog')
    ax.set_xlabel("Número d'itération")
    ax2.set_ylabel('MAE score (rouge)')
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(f'./out_data/precision_{mode}_{degree}.png')
    plt.show()

