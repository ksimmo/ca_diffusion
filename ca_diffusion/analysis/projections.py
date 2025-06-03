import numpy as np

def calculate_projections(data, corr_neighbourhood=3):
    data_calculated = {}

    data_calculated["max_intensity"] = np.amax(data)
    data_calculated["mean_intensity"] = np.mean(data)
    data_calculated["std_intensity"] = np.std(data)
    data_calculated["median_intensity"] = np.median(data)
    data_calculated["mad_intensity"] = np.median(data-data_calculated["median_intensity"])

    data_calculated["mean_image"] = np.mean(data, axis=0)
    data_calculated["std_image"] = np.std(data, axis=0)
    data_calculated["median_image"] = np.median(data, axis=0)
    data_calculated["max_image"] = np.amax(data, axis=0)

    #correlation image (lazy version)
    if corr_neighbourhood>0:
        corr_image_avg = np.zeros_like(data_calculated["mean_image"])
        corr_image_max = np.zeros_like(data_calculated["mean_image"])
        diff = data-np.expand_dims(data_calculated["mean_image"], axis=0)
        for i in range(corr_neighbourhood, data.shape[1]-corr_neighbourhood):
            for j in range(corr_neighbourhood, data.shape[2]-corr_neighbourhood):
                crop = diff[:,i-corr_neighbourhood:i+corr_neighbourhood+1,j-corr_neighbourhood:j+corr_neighbourhood+1]
                crop = crop.reshape(crop.shape[0],-1)
                crop_std = data_calculated["std_image"][i-corr_neighbourhood:i+corr_neighbourhood+1,j-corr_neighbourhood:j+corr_neighbourhood+1]
                crop_std = crop_std.flatten()

                center_i = crop_std.shape[0]//2

                #remove self-correlation
                crop = np.delete(crop, center_i, axis=1)
                crop_std = np.delete(crop_std, center_i, axis=0)

                corr = np.sum(diff[:,i,j].reshape(-1,1)*crop, axis=0)/(data_calculated["std_image"][i,j]*crop_std)
                corr_image_avg[i,j] = np.mean(corr)/data.shape[0]
                corr_image_max[i,j] = np.amax(corr/data.shape[0])
        data_calculated["corr_avg_image"] = corr_image_avg #move from [-1,1] to [0,2] to match normalization in main view
        data_calculated["corr_max_image"] = corr_image_max

    return data_calculated