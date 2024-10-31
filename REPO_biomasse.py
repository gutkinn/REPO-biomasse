import geopandas as gpd
import openeo
import json
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import xarray as xr
from helpers import *
from shapely import geometry, buffer

def main(in_pts_file,in_area_file,out_dir,year,month,it_mode,degree,download,point_extract):
    print("Établissant une connexion à openEO.")
    connection = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()

    if point_extract:
        print("Extraction des points d'entraînement d'openEO.")
        prodsites, points_out, bands = extract_training_features(in_pts_file,out_dir,year,connection,point_extract)
    else:
        print("Points d'entraînement déjà extraits.")
        prodsites, points_out, bands = extract_training_features(in_pts_file,out_dir,year,connection,point_extract)

    dict_dates = {}

    extr_data = pd.concat([prodsites['ID Point'],pd.read_json(points_out,convert_axes=False)],axis=1)
    extr_data = extr_data[['ID Point']+list(extr_data.keys()[1:].sort_values())]

    for key in extr_data.columns.values[1:]:
        dict_dates[key] = key[:10]
    extr_data.rename(columns=dict_dates,inplace=True)

    print("Organization des données d'entraînement.")
    X_df,y_df = organize_training_data(prodsites,extr_data,bands)

    print("Extraction de zone d'intérêt pour faire des prédictions.")
    in_data, area, crs, in_data_path = extract_area_prediction(in_area_file,out_dir,month,connection,bands,download)
    out_path = in_data_path.replace('100m','MODEL')

    print("Application du modèle au zone d'intérêt.")
    train_model(X_df,y_df,in_data,crs,out_path,area,it_mode,degree)

def extract_training_features(in_pts_file,out_dir,year,connection,download):
    # bandes satellitaires Sentinel-1 et Sentinel-2 dont nous avons besoin
    band_list_s1 = ['VV','VH']
    band_list_s2 = ['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B11','B12','SCL']
    bands = (band_list_s1,band_list_s2)

    # creneaux temporaires pour l'extraction des données
    t = [f'{year}-01-01',f'{year}-12-31']

    sites_csv = pd.read_csv(in_pts_file) # location des données biomasses dans les fichiers
    sites_points = [geometry.Point(x,y) for x,y in zip(sites_csv['Long (E)'],sites_csv['Lat (N)'])]
    prodsites = gpd.GeoDataFrame(sites_csv,crs='epsg:4326',geometry=sites_points)
    prodsites['geometry'] = buffer(prodsites.geometry.to_crs(3857),30).set_crs(3857).to_crs(4326)
    prodsites = prodsites.drop(columns='fid')

    ### ADD dates to all buffer locations
    all_dates = ['date_fev','date_avr','date_sep']
    all_biomass = ['biomass_fev','biomass_avr','biomass_sep']

    prodsites['dates'] = prodsites.apply(create_list, axis=1, args=(all_dates,))
    prodsites['biomass'] = prodsites.apply(create_list, axis=1, args=(all_biomass,True))

    bbox_simple = {"west":prodsites.geometry.to_crs(crs=3857).centroid.to_crs(4326).x.min()-0.0001,
               "south":prodsites.geometry.to_crs(crs=3857).centroid.to_crs(4326).y.min()-0.0001,
               "east":prodsites.geometry.to_crs(crs=3857).centroid.to_crs(4326).x.max()+0.0001,
               "north":prodsites.geometry.to_crs(crs=3857).centroid.to_crs(4326).y.max()+0.0001}

    geoms = json.loads(prodsites.explode(index_parts=True).geometry.to_json())

    #save points
    points_out = os.path.join('.',out_dir,'points_biomasse.json')

    if download:
        #extract training points
        extr_agg = extract_and_save_simple(band_list_s1,band_list_s2,t,bbox_simple,geoms,connection)
        job = extr_agg.create_job(
                title='extraction_points', out_format="JSON") # Changez le titre pour suivre les étapes d'extraction en ligne

        job.start_and_wait()

        for asset in job.get_results().get_assets():
            asset.download(points_out)
    
    return prodsites, points_out, bands

def organize_training_data(prodsites,extr_data,band_lists):
    (band_list_s1, band_list_s2) = band_lists

    all_indexes =['VV', 'VH', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
                'B8A', 'B11', 'B12', 'NDVI', 'NDWI', 'GRVI', 'GNDVI', 'NBR',
                'NBR2', 'NDI54', 'NDRE1', 'STI', 'MSI',  'SAVI', 'MSAVI', 'IRECI',
                'VV-VH_rolling','VV+VH_rolling']

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

    X_df = pd.DataFrame(index_list.T,columns=all_indexes,index=all_l1_time)
    y_df = pd.DataFrame(biomass_list,columns=['biomass'],index=all_l1_time)

    return X_df, y_df

def extract_area_prediction(in_area_file,out_dir,month,connection,bands,download):
    
    area = gpd.read_file(in_area_file).to_crs(epsg=4326)
    bounds = area.total_bounds

    in_data_path = os.path.join(out_dir,f"{in_area_file.split('/')[-1].replace('.gpkg','')}_100m.nc")

    # IMPORTANT : Il faut selectionner le mois entier pour avoir des prédictions de biomasse correctes
    t_extract = [f'2024-{month}-01', f'2024-{month}-30']

    bbox = {'west':bounds[0],
            'south':bounds[1],
            'east':bounds[2],
            'north':bounds[3],
            'crs':area.crs.to_string()}

    extr_area = extract_and_save_simple(bands[0],bands[1],t_extract,bbox,None,connection)

    if download:
        job = extr_area.create_job(
                title=f'extraction_{month}', out_format="NetCDF")
        job.start_and_wait()
        for asset in job.get_results().get_assets():
            asset.download(in_data_path)
    
    in_data = xr.open_dataset(in_data_path)
    add_indices(in_data,rolling=True)
    crs = in_data['crs'].attrs['crs_wkt']
    in_data = in_data.drop_vars(['crs']).mean(dim='t')

    return in_data, area, crs, in_data_path

def train_model(X_df,y_df,in_data,crs,out_path,area,it_mode,degree=0):
    print(f'degré: {degree}')
    if it_mode == 'RF':
        mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode='RF')
        model = get_best_model(X_df[remaining_feats[max(r2_scores,key=r2_scores.get)]],y_df,'RF')
        predict_and_save(model,in_data,crs,out_path,area.geometry,it_mode = 'RF',)
        
    if it_mode == 'PLS':
        mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode='PLS')
        model = get_best_model(X_df[remaining_feats[21]],y_df,'PLS',degree)
        coef = dict(zip(remaining_feats[21],model.coef_[0]))
        predict_and_save(model,in_data,crs,out_path,area.geometry,it_mode='PLS',coef=coef)

    if it_mode == 'LR' and degree == 1:
        mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode='LR',degree=degree)
        model = get_best_model(X_df[remaining_feats[max(r2_scores,key=r2_scores.get)]],y_df,'LR',degree)
        coef = dict(zip(remaining_feats[max(r2_scores,key=r2_scores.get)],model.coef_[0]))
        predict_and_save(model,in_data,crs,out_path,area.geometry,it_mode='LR',coef=coef,degree=degree)

    if it_mode == 'LR' and degree > 1:
        mae_scores,r2_scores,remaining_feats,preds = iterate_train_test(X_df,y_df,split=0.25,it_mode='LR',degree=degree)
        model,poly = get_best_model(X_df[remaining_feats[max(r2_scores,key=r2_scores.get)]],y_df,'LR',degree)
        coef = dict(zip(poly.get_feature_names_out(),model.coef_[0]))
        predict_and_save(model,in_data,crs,out_path,area.geometry,it_mode='LR',coef=coef,degree=degree)


if __name__ == "__main__":
    import argparse
    import json

    with open(r'./test_config.json') as cfile:
        config = json.loads(cfile.read())
    
    main(config['in_points'], config['in_area'], config['out_dir'],
        config['train_year'], config['predict_month'], config['mode'],
        config['degree'], config['download'], config['point_extract'])
