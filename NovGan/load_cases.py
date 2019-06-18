from copy import deepcopy as deepcopy


def load_cases(link_modes=["alpha", "exp", "pow", "sum"],
               loss_bases=None,
               alphas=[1.,2.,10.],
               offsets=[0.,1.,7.5],
               powers=[1,2,10],
               mults=[1.,2.],
               sqrts=[1,3]):

    if loss_bases is None:
        loss_bases = ["Goodfellow", "Wasserstein", "Pearson"]
    output = list()
    dico = {"link_mode": None,
            "loss_base": None,
            "alpha": None,
            "offset": None,
            "power": None,
            "mult": None,
            "sqrt": None}

    for link_mode in link_modes:
        for loss_base in loss_bases:
            if link_mode == "alpha":
                for alpha in alphas:
                    for offset in offsets:
                        dic = deepcopy(dico)
                        dic["link_mode"] = link_mode
                        dic["loss_base"] = loss_base
                        dic["alpha"] = alpha
                        dic["offset"] = offset
                        output.append(dic)
            elif link_mode == "exp":
                dic = deepcopy(dico)
                dic["link_mode"] = link_mode
                dic["loss_base"] = loss_base
                output.append(dic)
            elif link_mode == "pow":
                for power in powers:
                    dic = deepcopy(dico)
                    dic["link_mode"] = link_mode
                    dic["loss_base"] = loss_base
                    dic["power"] = power
                    output.append(dic)
            elif link_mode == "sum":
                for mult in mults:
                    for sqrt in sqrts:
                        dic = deepcopy(dico)
                        dic["link_mode"] = link_mode
                        dic["loss_base"] = loss_base
                        dic["mult"] = mult
                        dic["sqrt"] = sqrt
                        output.append(dic)

    return output


def load_save_name(case):
    loss_base = "loss_base:" + str(case["loss_base"]) + "-"
    link_mode = "link_mode:" + str(case["link_mode"]) + "-"
    alpha = "alpha:" + str(case["alpha"]) + "-"
    offset = "offset:" + str(case["offset"]) + "-"
    power = "power:" + str(case["power"]) + "-"
    mult = "mult:" + str(case["mult"]) + "-"
    sqrt = "sqrt:" + str(case["sqrt"])

    output = loss_base + link_mode + alpha + offset + power + mult +sqrt
    return output

