import pickle
import time

import flopy
import flopy.utils.binaryfile as bf
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from radom_ground_env import RandomGroundEnv


def generate_mock_data():
    param = RandomGroundEnv()
    model_path = "mock_data"
    model_name = "mdata_0"
    mf_obj = flopy.modflow.Modflow(
        model_name,
        model_ws=model_path,
    )
    discretization = flopy.modflow.ModflowDis(
        mf_obj,
        param.num_layer, 
        param.num_row, 
        param.num_col,
        delr=param.delta_row,
        delc=param.delta_col,
        top=param.z_top,
        botm=param.bottom_arr[1:],
    )
    basic_info = flopy.modflow.ModflowBas(
        mf_obj,
        ibound=param.boundary,
        strt=param.starting_head,
    )
    layer_property_flow = flopy.modflow.ModflowLpf(
        mf_obj,
        hk=param.layer_property,
    )
    well_obj = flopy.modflow.ModflowWel(
        mf_obj,
        stress_period_data=param.well_stress_data
    )

    # output config
    stress_period_oc = {(0, 0): ["print head", "print budget", "save head", "save budget"]}
    output_config = flopy.modflow.ModflowOc(
        mf_obj,
        stress_period_data=stress_period_oc,
        compact=True,
    )
    pcg = flopy.modflow.ModflowPcg(mf_obj)

    mf_obj.write_input()
    succ_flag, buff = mf_obj.run_model(silent=True)
    if not succ_flag:
        raise RuntimeError("ERROR")
    
    # 提取水头数据
    head_file = bf.HeadFile(f'./{model_path}/{model_name}.hds')
    head = head_file.get_data(totim=1.0)
    param.head = head

    return param

    df = pd.DataFrame(head[0])
    plt.figure(figsize=(11,8))
    plt.imshow(df)
    plt.colorbar()
    plt.show()


def batch_generate(amount: int):
    data = [
        generate_mock_data() for _ in tqdm(
            range(amount),
            leave=False,
            ncols=90,
        )
    ]
    file_path = 'mock_data'
    file_name = f'./{file_path}/mock_data-{int(time.time())}.pkl'
    with open(file_name, 'wb') as fp:
        pickle.dump(data, fp)
    return


if __name__ == '__main__':
    # generate_mock_data()
    batch_generate(10000)
