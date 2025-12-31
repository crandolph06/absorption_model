from models import Qual, Upgrade

def rap_state_code(pilots):
    groups = {
        "MQT": [p for p in pilots if p.upgrade == Upgrade.MQT],
        "Wingmen": [p for p in pilots if p.qual == Qual.WG and p.upgrade != Upgrade.MQT],
        "Flight Leads": [p for p in pilots if p.qual == Qual.FL],
        "Instructors": [p for p in pilots if p.qual == Qual.IP]
    }

    for group in groups:
        avg_sorties = sum(p.sortie_monthly for p in group) / len(group)
        avg_blue_sorties = sum(p.sortie_blue_monthly for p in group) / len(group)
        avg_red_sorties = sum(p.sortie_red_monthly for p in group) / len(group)

        #TODO need to assess monthly RAP for groups and produce RAP state label.

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

rap_code = 
rap_label = rap_state_label(rap_code)