import os 

def main(in_points,in_area,out_dir,train_year,predict_month,mode,degree,download):
    print(in_points, os.path.exists(in_points))
    print(in_area, os.path.exists(in_area))
    print(out_dir,os.path.exists(out_dir))
    print(train_year, type(train_year))
    print(predict_month, type(predict_month))
    print(mode,type(mode))
    print(degree,type(degree))
    if download:
        print(download)

if __name__ == "__main__":
    import json

    with open(r'./test_config.json') as cfile:
        config = json.loads(cfile.read())
    print(config)
    main(config['in_points'], config['in_area'], config['out_dir'],
        config['train_year'], config['predict_month'], config['mode'],
        config['degree'], config['download'])