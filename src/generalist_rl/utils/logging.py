from typing import Dict


def get_logging_str_from_dict(d: Dict, width:int=80, pad:int=35) -> str:
    logging_str = ""
    logging_str += f"""{'#' * width}\n"""
    for k, v in d.items():
        if type(v) == float:
            logging_str += f"{k:>{pad}}: {v:.6f}"
        elif type(v) == int:
            logging_str += f"{k:>{pad}}: {v}"
        else:
            print(f"unknown type for key {k}: {type(v)}")
            logging_str += f"{k:>{pad}}: {v}"
        logging_str += "\n"
    logging_str += f"""{'#' * width}\n"""
    return logging_str
