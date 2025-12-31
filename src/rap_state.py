from models import Qual, Upgrade

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

    for group in groups:
        avg_sorties = sum(p.sortie_monthly for p in group) / len(group)
        avg_blue_sorties = sum(p.sortie_blue_monthly for p in group) / len(group)
        avg_red_sorties = sum(p.sortie_red_monthly for p in group) / len(group)

        if group.key() == "WG":
            rap_req = 9
            bit_mask = 1
        elif group.key() == "FL":
            rap_req = 8
            bit_mask = 2
        elif group.key() == "IP":
            rap_req = 8
            bit_mask = 4

        if avg_sorties >= rap_req: 
            rap_dict[group.key()] = [bit_mask, avg_sorties] # rap_dict["WG"] = [1, 9.5]
        else:
            rap_dict[group.key()] = [0, avg_sorties] # rap_dict["WG"] = [0, 5.5]

        if avg_blue_sorties >= rap_req:
            blue_rap_dict[group.key()] = [bit_mask, avg_blue_sorties] # blue_rap_dict["FL"] = [2, 9.5]
        else: 
            blue_rap_dict[group.key()] = [0, avg_blue_sorties] # blue_rap_dict["WG"] = [0, 3.5]

        red_percentage = avg_red_sorties / avg_sorties
        red_dict[group.key()] = red_percentage # red_dict["WG"] = 45.5

    return rap_dict, blue_rap_dict, red_dict

def rap_state_code(rap_dict):
    rap_code = 0

    for k,v in rap_dict:
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
