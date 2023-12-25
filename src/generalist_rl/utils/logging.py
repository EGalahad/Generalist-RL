from typing import Dict
def get_logging_str_from_dict(d: Dict) -> str:
    logging_str = ""
    logging_str += "========================================\n"
    for k, v in d.items():
        if type(v) == float:
            logging_str += f"{k}: {v:.6f}"
        elif type(v) == int:
            logging_str += f"{k}: {v}"
        else:
            logging_str += f"{k}: {v}"
        logging_str += "\n"
    logging_str += "========================================\n"
    return logging_str