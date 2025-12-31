from src.models import Qual, Upgrade

def rap_assess(pilots):
    groups = {
        "MQT": [p for p in pilots if p.upgrade == Upgrade.MQT],
        "WG": [p for p in pilots if p.qual == Qual.WG and p.upgrade != Upgrade.MQT],
        "FL": [p for p in pilots if p.qual == Qual.FL],
        "IP": [p for p in pilots if p.qual == Qual.IP]
    }

    rap_dict = {}
    blue_rap_dict = {}
    red_dict = {}

    for group_name, group_pilots in groups.items():
        if not group_pilots:
            rap_dict[group_name] = [0 ,0]
            blue_rap_dict[group_name] = [0, 0]
            red_dict[group_name] = 0
            continue

        avg_sorties = sum(p.sortie_monthly for p in group_pilots) / len(group_pilots)
        avg_blue_sorties = sum(p.sortie_blue_monthly for p in group_pilots) / len(group_pilots)
        avg_red_sorties = sum(p.sortie_red_monthly for p in group_pilots) / len(group_pilots)

        if group_name == "WG":
            rap_req, bit_mask = 9, 1
        elif group_name == "FL":
            rap_req, bit_mask = 8, 2
        elif group_name == "IP":
            rap_req, bit_mask = 8, 4
        elif group_name == "MQT":
            rap_req, bit_mask = 0, 0
            
        rap_dict[group_name] = [bit_mask if avg_sorties < rap_req else 0, avg_sorties] # rap_dict["WG"] = [1, 9.5]
        blue_rap_dict[group_name] = [bit_mask if avg_blue_sorties < rap_req else 0, avg_blue_sorties] # blue_rap_dict["FL"] = [2, 9.5]
        red_dict[group_name] = avg_red_sorties / avg_sorties if avg_sorties > 0 else 0 # red_dict["WG"] = 45.5

    return rap_dict, blue_rap_dict, red_dict

def rap_state_code(rap_dict):
    rap_code = 0

    for k,v in rap_dict.items():
        if k in ["WG", "FL", "IP"]:
            rap_code += v[0]

    return rap_code

def rap_state_label(code):
    labels = {
        0: "All Make RAP",
        1: "WG Shortfall",
        2: "FL Shortfall",
        3: "WG + FL Shortfall",
        4: "IP Shortfall",
        5: "WG + IP Shortfall",
        6: "FL + IP Shortfall",
        7: "WG + FL + IP Shortfall",
    }
    return labels[code]
