import pandas as pd
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import save_img
import os.path
import errno
from tqdm import tqdm

def resize_images(filename, target_size):
    # resize images to new target size. Input is pandas dataframe of full dataset before splitting of train/val/test
    df = pd.read_csv(filename)
    path = df['Path'].values
    path_compr = df['Path_compr'].values
    rows_to_remove = []
    for idx in tqdm(range(len(path))):
        try:
            if path[idx][0] != '/':
                path[idx] = '/' + path[idx]
                path_compr[idx] = '/' + path_compr[idx]
            if not os.path.exists(path_compr[idx]):
                try:
                    os.makedirs(os.path.dirname(path_compr[idx]), exist_ok=True)
                except OSError as exc:  # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        print("race condition")
                        raise
                array = img_to_array(load_img(path[idx], target_size = target_size)))
                save_img(path_compr[idx], array)
        except:
            rows_to_remove.append(idx)
    df['Path_compr'] = path_compr
    df=df.drop(df.index[rows_to_remove])
    # save new version in which instances without valid image are removed
    df.to_csv(filename)



if __name__ == '__main__':
    resize_images(filename, target_size = (224,224))
