import argparse
import yaml, os
from jinja2 import Template
from pathlib import Path
import shutil

def get_config(path):
    f = open(path, encoding='utf-8')
    y = yaml.load(f, yaml.FullLoader)
    return y

def get_render_config(path):
    # 获取YAML文件中的参数文件
    data = get_config(path)
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # 加载带有模板的YAML文件
    with open('config.yml', 'r', encoding='utf-8') as file:
        template = Template(file.read())
    # 定义参数的值
    parameters = {
        'parameter1': 'value1',
        'p2': BASE_DIR,
        'dc': data['data_city'],
        'num': len(data['DATA_DICT'][data['data_city']]['color']),
        'tr': data['train_rate'],
        'ep': data['epoch'],
        'bs': data['batchsize'],
        'expo_result': data['expo_result'],
        'parameters': data['parameters'],
        'mn': data['model_name'],
        'FN': data['FILE_NUM'],
        
        'ne': data['dqtl']['num_epochs'],
        'ps': data['dqtl']['pic_size'],
    }
    # 渲染模板并输出结果
    rendered_yaml = template.render(**parameters)
    y = yaml.load(rendered_yaml, yaml.FullLoader)
    y = get_dump_config(y)
    return y

def get_dump_config(y):
    if os.path.exists(y['RESULT']) == 0:  # 确定实验结果图存放位置
        os.makedirs(y['RESULT'])  # 如果不存在文件夹，新建文件夹保存图片数据
    filenum = 0
    RESULT_excel = y['RESULT'] + y['model_name'] + "__" + str(filenum) + '_result.xlsx'
    RESULT_output = y['RESULT'] + y['model_name'] + "__" + str(filenum) + '_output/'
    if not y['train']['index'] == 0:
        while os.path.exists(RESULT_excel) or os.path.exists(RESULT_output):
            filenum += 1
            RESULT_excel = y['RESULT'] + y['model_name'] + "__" + str(filenum) + '_result.xlsx'
            RESULT_output = y['RESULT'] + y['model_name'] + "__" + str(filenum) + '_output/'
        y['FILE_NUM'] = filenum
        if y['delete']:
            len = filenum
            for i in range(len):
                # print(i)
                num = len - i - 1
                current_directory = y['RESULT'] + y['model_name'] + "__" + str(num) + '_output'
                # has_files = any(os.path.isfile(os.path.join(current_directory, f)) for f in os.listdir(current_directory))
                has_xlsx = os.path.isfile(y['RESULT'] + y['model_name'] + "__" + str(num) + '_result.xlsx')
                # print(num, os.path.exists(current_directory), os.path.isdir(current_directory))
                if os.path.exists(current_directory) and os.path.isdir(current_directory) and not has_xlsx:
                    # 使用shutil.rmtree删除文件夹
                    shutil.rmtree(current_directory)
                    filenum = num
                    y['FILE_NUM'] = filenum
                    # print(num)
                if num == 0:
                    break
    # data = yaml.safe_load(y)

        # y['dqtl']['WEIGHTS'] = 'dqtl_'+y['data_city']+'/'+str(y['dqtl']['num_epochs'])+\
        # #                        '_'+str(y['dqtl']['pic_size'])+'_'+str(filenum)+'/'
        # y['dqtl']['WEIGHTS'] = 'dqtl_' + y['data_city'] + '/' + str(y['dqtl']['num_epochs']) + \
        #                        '_' + str(y['dqtl']['pic_size']) + '_' + '0' + '/'
    else:
        filenum = y['FILE_NUM']
    y['RESULT_excel'] = RESULT_excel
    y['RESULT_output'] = y['RESULT'] + y['model_name'] + "__" + str(filenum) + '_output/'
    y['schedule']['lr'] = float(y['schedule']['lr'])
    y['schedule']['base_lr'] = float(y['schedule']['base_lr'])
    y['Categories_Number'] = int(y['Categories_Number'])
    y['dqtl']['lr'] = float(y['dqtl']['lr'])
    y['dqtl']['tao'] = float(y['dqtl']['tao'])
    y['dqtl']['epsilon'] = float(y['dqtl']['epsilon'])
    
    yaml_str = yaml.dump(y)
    y = yaml.safe_load(yaml_str)
    if not os.path.exists(y['RESULT_output']) and y['train']['save_best']:
        os.makedirs(y['RESULT_output'])
    return y


if __name__ == "__main__":
    y = get_render_config("./config.yml")
    print(y)
