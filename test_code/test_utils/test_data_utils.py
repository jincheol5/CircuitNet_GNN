import argparse
from utils import DataUtils

def test_fn(**kwargs):
    match kwargs["test_num"]:
        case 1:
            """
            DataUtils.load_timing_features_data
            """
            nodes_data=DataUtils.load_timing_features_data(
                file_name=kwargs["file_name"],
                data_type=f"nodes"
            )
            pin_positions_data=DataUtils.load_timing_features_data(
                file_name=kwargs["file_name"],
                data_type=f"pin_positions"
            )
            print(f"nodes data:")
            print(type(nodes_data))
            print(nodes_data.files)
            print(nodes_data["nodes"][:5],end="\n\n")

            print(f"pin_positions data:")
            print(type(pin_positions_data))
            print(pin_positions_data.files)
            print(type(pin_positions_data["pin_positions"])) # 0차원 object array: NumPy 배열이라는 껍데기 안에 Python 객체 하나를 통째로 넣어둔 형태
            print(pin_positions_data["pin_positions"].shape) # 0차원 object array
            print(type(pin_positions_data["pin_positions"].item()),end="\n\n") # dict
            position_dict=pin_positions_data["pin_positions"].item()
            print(f"key ex: {list(position_dict.keys())[:5]}")
            position_value_ex=position_dict[f"FE_OCPC10204_testmode_i/I"]
            print(f"pin_position value ex: {position_value_ex}")
            # print(f"{kwargs['file_name']} position: {position_dict[kwargs['file_name']]}")

if __name__=="__main__":
    """
    Execute test_fn
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--test_num",type=int,default=1)
    parser.add_argument("--file_name",type=str,default=f"8-RISCY-a-1-c2-u0.7-m1-p8-f0")
    args=parser.parse_args()
    test_config={
        "test_num":args.test_num,
        "file_name":args.file_name
    }
    test_fn(**test_config)