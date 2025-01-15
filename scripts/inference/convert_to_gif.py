import os
import argparse
import numpy as np

import imageio

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, help="Path to results")
    opt = parser.parse_args()

    if not os.path.exists(opt.path):
        print("Path {} does not exists!".format(opt.path))
        exit()

    

    obj = np.load(opt.path)

    video = obj["arr_0"]
    rec = obj["arr_1"]
    video_mean = obj["arr_2"]
    rec_mean = obj["arr_3"]

    frames = []
    for i in range(video.shape[0]):
        fr = np.concatenate([video_mean, video[i], rec[i], rec_mean], axis=-1)
        fr = (fr+1.0)*0.5*255.0
        #fr = np.expand_dims(fr, axis=-1)
        frames.append(fr.astype(np.uint8))
    imageio.mimsave(os.path.join(os.path.dirname(opt.path), "output.gif".format(i)), frames)
